import asyncio
import datetime
import json
import logging
import os
import subprocess
import sys
from collections import deque
from typing import cast

import MetaTrader5 as mt5
import numpy as np
import pandas as pd  # Wajib ada untuk proses data
import redis.asyncio as redis
import torch

from cloud_manager import cloud_manager
from config import settings
from database import get_db
from features import processor  # Import processor untuk scaling data
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

# brain.py


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-V1")


class Brain:
    def __init__(self):
        # Buffer dilebihkan sedikit dari SEQ_LEN agar indikator teknikal (RSI/EMA) bisa terhitung
        self.buffer = deque(maxlen=settings.SEQ_LEN + 50)
        self.model = None
        self.r_brain = None

        # State Self-Healing
        self.consecutive_losses = 0
        self.is_training_mode = False

        # State Trading
        self.is_pending_order = False
        self.pending_start_time = None

    async def init(self):
        await self.load_model()
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r_brain = cast(redis.Redis, redis.Redis(**kwargs))
        asyncio.create_task(self.listen_events())

    async def load_model(self):
        self.model = TimeSeriesTransformer(input_dim=4)
        if not os.path.exists(settings.MODEL_FILE):
            await asyncio.to_thread(cloud_manager.download_model)

        try:
            self.model.load_state_dict(torch.load(settings.MODEL_FILE))
            self.model.eval()  # Set mode evaluasi (bukan training)
            logger.info("🧠 AI Model Loaded Successfully.")
        except:
            logger.warning("⚠️ Model not found or corrupt. Running Logic Only.")

    async def listen_events(self):
        assert self.r_brain is not None
        pubsub = self.r_brain.pubsub()
        await pubsub.subscribe(settings.CHANNEL_CONFIRMATION, settings.CHANNEL_SYSTEM)

        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    channel = msg["channel"]

                    if (
                        channel == settings.CHANNEL_SYSTEM
                        and data.get("event") == "TRAINING_COMPLETED"
                    ):
                        logger.info("🎓 Training Done. Reloading Brain...")
                        await self.load_model()
                        self.consecutive_losses = 0
                        self.is_training_mode = False

                    elif channel == settings.CHANNEL_CONFIRMATION:
                        self.is_pending_order = False
                        self.pending_start_time = None

                        if data.get("status") == "CLOSED":
                            pnl = float(data.get("pnl", 0))
                            if pnl < 0:
                                self.consecutive_losses += 1
                                if (
                                    self.consecutive_losses
                                    >= settings.RETRAIN_ON_LOSS_COUNT
                                    and settings.AUTO_RETRAIN
                                ):
                                    await self.trigger_retrain()
                            else:
                                self.consecutive_losses = 0
                except Exception as e:
                    logger.error(f"Event Error: {e}")

    async def trigger_retrain(self):
        assert self.r_brain is not None
        logger.warning("😡 Max Loss Hit. Triggering Auto-Retrain...")
        self.is_training_mode = True
        await self.r_brain.publish(
            settings.CHANNEL_SYSTEM, json.dumps({"event": "TRAINING_START"})
        )
        subprocess.Popen([sys.executable, "train.py"])

    async def manage_positions(self, c):
        active_pos = await state_manager.get_active_position()
        if not active_pos or self.is_pending_order:
            return

        current_price = float(c["close"])

        # Trailing Stop Logic
        if active_pos["side"] == "BUY":
            entry = float(active_pos["entry_price"])
            sl = float(active_pos["sl"])
            if (current_price - entry) / entry > 0.01:
                new_sl = current_price * 0.995
                if new_sl > sl:
                    active_pos["sl"] = new_sl
                    await state_manager.set_active_position(
                        active_pos["symbol"],
                        "BUY",
                        entry,
                        active_pos["lot"],
                        active_pos["tp"],
                        new_sl,
                    )
                    logger.info(f"📈 Trailing SL moved to: {new_sl}")

        # Check TP/SL locally (Backup jika broker belum close)
        high = float(c["high"])
        low = float(c["low"])
        tp = float(active_pos["tp"])
        sl = float(active_pos["sl"])
        signal_close = False

        if active_pos["side"] == "BUY":
            if low <= sl:
                signal_close = True
            elif high >= tp:
                signal_close = True
        elif active_pos["side"] == "SELL":
            if high >= sl:
                signal_close = True
            elif low <= tp:
                signal_close = True

        if signal_close:
            logger.info(f"🚨 Close Signal Triggered locally")
            self.is_pending_order = True
            self.pending_start_time = datetime.datetime.now()
            # Executor akan menangani close order sebenarnya di broker

    def is_market_open(self):
        return datetime.datetime.now().weekday() < 5

    def warmup_data(self):
        """Isi buffer dengan data historis agar indikator akurat"""
        if not mt5.initialize():
            logger.error("❌ MT5 Init Failed for Warmup")
            return

        symbol = settings.ACTIVE_SYMBOLS[0]  # Ambil simbol aktif
        logger.info(f"🔥 Warming up brain with data for {symbol}...")

        # Ambil 200 candle terakhir (M1 atau sesuai timeframe)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 200)

        if rates is not None:
            for rate in rates:
                # Format harus sama dengan stream_manager
                c = {
                    "symbol": symbol,
                    "close": float(rate["close"]),
                    "open": float(rate["open"]),
                    "high": float(rate["high"]),
                    "low": float(rate["low"]),
                    "volume": float(rate["tick_volume"]),
                    "timestamp": str(rate["time"]),
                }
                self.buffer.append(c)
            logger.info(f"✅ Brain Warmed up! Buffer size: {len(self.buffer)}")
        else:
            logger.warning("⚠️ Failed to get history. Brain starts COLD (Risk high!)")

    async def run(self):
        await self.init()
        await streamor.connect()
        self.warmup_data()

        logger.info("🧠 Brain V1 Running (AI Inference Active)...")

        while True:
            if self.is_training_mode:
                await asyncio.sleep(5)
                continue

            if not self.is_market_open():
                await asyncio.sleep(300)
                continue

            candles = await streamor.consume_market_data()
            if not candles:
                await asyncio.sleep(0.1)
                continue

            if self.is_pending_order and self.pending_start_time:
                if (
                    datetime.datetime.now() - self.pending_start_time
                ).total_seconds() > 60:
                    self.is_pending_order = False
                    self.pending_start_time = None

            for c in candles:
                self.buffer.append(c)
                await self.manage_positions(c)

                if await state_manager.get_active_position() or self.is_pending_order:
                    continue

                # --- LOGIKA AI (REAL INFERENCE) ---
                action = "HOLD"

                if (
                    self.model and len(self.buffer) >= settings.SEQ_LEN + 20
                ):  # Buffer lebih utk hitung indikator
                    try:
                        # 1. Konversi Buffer ke DataFrame
                        df = pd.DataFrame(list(self.buffer))

                        # 2. Proses Data (Hitung RSI, EMA, lalu Scaling)
                        # Ini menggunakan logika yang sama persis dengan train.py
                        _, scaled_data = processor.process(df)

                        # 3. Cek panjang data setelah dipotong indikator (NaN drop)
                        if len(scaled_data) >= settings.SEQ_LEN:
                            # Ambil SEQ_LEN terakhir
                            input_seq = scaled_data[-settings.SEQ_LEN :]

                            # 4. Konversi ke Tensor [Batch, Seq, Feature]
                            tensor = torch.FloatTensor(input_seq).unsqueeze(0)

                            # 5. Prediksi
                            with torch.no_grad():
                                logits = self.model(tensor)
                                # Mapping: 0=HOLD, 1=BUY, 2=SELL (Sesuai train.py)
                                pred = torch.argmax(logits, dim=1).item()

                                if pred == 1:
                                    action = "BUY"
                                elif pred == 2:
                                    action = "SELL"

                    except Exception as e:
                        logger.error(f"Prediction Error: {e}")

                # Kirim Sinyal jika ada
                if action != "HOLD":
                    logger.info(f"🚀 AI Signal Generated: {action}")
                    self.is_pending_order = True
                    self.pending_start_time = datetime.datetime.now()

                    await streamor.push_signal(
                        {
                            "action": action,
                            "symbol": settings.ACTIVE_SYMBOLS[0],
                            "entry_price": c["close"],
                            "timestamp": str(datetime.datetime.now()),
                        }
                    )


if __name__ == "__main__":
    asyncio.run(Brain().run())
