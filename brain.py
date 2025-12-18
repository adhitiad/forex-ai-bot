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
import pandas as pd
import redis.asyncio as redis
import torch
import torch.nn.functional as F

from cloud_manager import cloud_manager
from config import settings
from database import get_db
from features import processor
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-V1")


class Brain:
    def __init__(self):
        self.buffer = deque(maxlen=settings.SEQ_LEN + 50)
        self.model = None
        self.r_brain = None
        self.consecutive_losses = 0
        self.is_training_mode = False
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
            self.model.eval()
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

        high = float(c["high"])
        low = float(c["low"])
        tp = float(active_pos["tp"])
        sl = float(active_pos["sl"])
        signal_close = False

        if active_pos["side"] == "BUY":
            if low <= sl or high >= tp:
                signal_close = True
        elif active_pos["side"] == "SELL":
            if high >= sl or low <= tp:
                signal_close = True

        if signal_close:
            logger.info(f"🚨 Close Signal Triggered locally")
            self.is_pending_order = True
            self.pending_start_time = datetime.datetime.now()

    def is_market_open(self):
        return True

    def warmup_data(self):
        if not mt5.initialize():
            logger.error("❌ MT5 Init Failed for Warmup")
            return
        symbol = settings.ACTIVE_SYMBOLS[0]
        logger.info(f"🔥 Warming up brain with data for {symbol}...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 200)
        if rates is not None:
            for rate in rates:
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

    def generate_explanation(self, rsi, price, ema):
        """Membuat alasan logis (Keterangan) berdasarkan indikator"""
        reasons = []

        # Analisis RSI
        if rsi > 70:
            reasons.append(f"RSI Overbought({rsi:.0f})")
        elif rsi < 30:
            reasons.append(f"RSI Oversold({rsi:.0f})")
        else:
            reasons.append(f"RSI Neutral({rsi:.0f})")

        # Analisis Trend (Price vs EMA)
        if price > ema:
            reasons.append("Trend UP")
        else:
            reasons.append("Trend DOWN")

        return " + ".join(reasons)

    async def run(self):
        await self.init()
        await streamor.connect()
        self.warmup_data()

        logger.info("🧠 Brain V1 Running (AI Inference Active)...")

        while True:
            if self.is_training_mode:
                await asyncio.sleep(5)
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

                action = "HOLD"
                prob_pct = 0.0
                explanation = "Analyzing..."
                order_type = "MARKET"  # Default untuk AI Brain V1 adalah Market Order

                if self.model and len(self.buffer) >= settings.SEQ_LEN + 20:
                    try:
                        df = pd.DataFrame(list(self.buffer))
                        df_features, scaled_data = processor.process(df)

                        if len(scaled_data) >= settings.SEQ_LEN:
                            input_seq = scaled_data[-settings.SEQ_LEN :]
                            tensor = torch.FloatTensor(input_seq).unsqueeze(0)

                            with torch.no_grad():
                                logits = self.model(tensor)
                                probs = F.softmax(logits, dim=1)
                                top_p, top_class = torch.max(probs, dim=1)

                                pred = top_class.item()
                                prob_pct = top_p.item() * 100

                                if pred == 1:
                                    action = "BUY"
                                elif pred == 2:
                                    action = "SELL"

                            # Generate Keterangan
                            last_row = df_features.iloc[-1]
                            rsi_val = last_row.get("RSI_14", 50)
                            ema_val = last_row.get("EMA_20", 0)
                            close_val = last_row.get("close", 0)

                            explanation = self.generate_explanation(
                                rsi_val, close_val, ema_val
                            )

                    except Exception as e:
                        logger.error(f"Prediction Error: {e}")

                if action != "HOLD":
                    # Tetapkan tipe order secara eksplisit untuk log dan payload
                    order_type_display = "NOW (Market)"

                    # --- FORMAT LOG LENGKAP ---
                    log_msg = (
                        f"\n📊 AI SIGNAL REPORT:\n"
                        f"• Prices       : {c['close']}\n"
                        f"• Action       : {action}\n"
                        f"• Tipe         : {order_type_display}\n"  # <--- Tipe Order Ditampilkan
                        f"• Prob         : {prob_pct:.2f}%\n"
                        f"• Kepercayaan  : {prob_pct:.2f}%\n"
                        f"• Ket          : {explanation}"
                    )
                    logger.info(log_msg)
                    # --------------------------

                    self.is_pending_order = True
                    self.pending_start_time = datetime.datetime.now()

                    # KIRIM BALIK / PUSH SIGNAL SEGERA
                    # Menambahkan field 'type' agar Fusion Engine/Executor tahu ini eksekusi langsung
                    payload = {
                        "action": action,
                        "type": "MARKET",  # <--- Dikirim di payload
                        "symbol": settings.ACTIVE_SYMBOLS[0],
                        "entry_price": c["close"],
                        "timestamp": str(datetime.datetime.now()),
                        "confidence": prob_pct,
                        "reason": explanation,
                        "source": "BRAIN_V1",
                    }
                    await streamor.push_signal(payload)
                    logger.info("🚀 Signal Pushed Downstream Immediately.")


if __name__ == "__main__":
    asyncio.run(Brain().run())
