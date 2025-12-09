import asyncio
import datetime
import json
import logging
from collections import deque
from typing import Optional

import redis.asyncio as redis
import torch

from config import settings
from database import TradeLog, get_db
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain-V2")


class Brain:
    def __init__(self):
        self.buffer = deque(maxlen=300)
        self.model = None
        self.is_pending_order = False
        self.pending_start_time = None
        self.db_gen = get_db()
        self.db = next(self.db_gen)
        self.r_brain: Optional[redis.Redis] = None

    async def init(self):
        # Load Model (Safe Fail)
        self.model = TimeSeriesTransformer(input_dim=4)
        try:
            self.model.load_state_dict(torch.load(settings.MODEL_FILE))
            self.model.eval()
        except:
            logger.warning("⚠️ Model not found, running logic only")

        self.r_brain = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        asyncio.create_task(self.listen_confirmations())

    async def listen_confirmations(self):
        if self.r_brain is None:
            logger.error("❌ Redis connection not initialized!")
            return

        pubsub = self.r_brain.pubsub()
        await pubsub.subscribe(settings.CHANNEL_CONFIRMATION)
        logger.info("👂 Listening Confirmations...")

        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    await self.handle_confirmation(data)
                except Exception as e:
                    logger.error(f"Confirm Error: {e}")

    async def handle_confirmation(self, data):
        action = data["original_action"]
        price = data["filled_price"]
        symbol = data["symbol"]

        logger.info(f"🤝 Confirmed: {action} @ {price}")
        self.is_pending_order = False
        self.pending_start_time = None

        if action == "BUY":
            tp = price * (1 + settings.TAKE_PROFIT_PIPS)
            sl = price * (1 - settings.STOP_LOSS_PIPS)

            # Save DB & State
            trade = TradeLog(
                symbol=symbol,
                action="BUY",
                entry_price=price,
                tp_price=tp,
                sl_price=sl,
                status="OPEN",
            )
            self.db.add(trade)
            self.db.commit()
            await state_manager.set_active_position(symbol, "BUY", price, 0.001, tp, sl)

        elif action == "SELL":
            await state_manager.clear_active_position()

    async def manage_positions(self, c):
        active_pos = await state_manager.get_active_position()
        if not active_pos or self.is_pending_order:
            return

        current_price = float(c["close"])

        # --- TRAILING STOP ---
        if active_pos["side"] == "BUY":
            entry = float(active_pos["entry_price"])
            sl = float(active_pos["sl"])
            if (current_price - entry) / entry > 0.01:  # Profit > 1%
                new_sl = current_price * 0.995  # SL naik
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
                    logger.info(f"📈 Trailing SL: {new_sl}")

        # --- CHECK TP/SL ---
        high = float(c["high"])
        low = float(c["low"])
        tp = float(active_pos["tp"])
        sl = float(active_pos["sl"])
        signal_close = False
        reason = ""

        if active_pos["side"] == "BUY":
            if low <= sl:
                signal_close, reason = True, "SL Hit"
            elif high >= tp:
                signal_close, reason = True, "TP Hit"

        if signal_close:
            logger.info(f"🚨 Closing Position: {reason}")
            self.is_pending_order = True
            self.pending_start_time = datetime.datetime.now()

            await streamor.push_signal(
                {
                    "action": "CLOSE_BUY",
                    "symbol": settings.ACTIVE_SYMBOL,
                    "entry_price": current_price,
                    "timestamp": str(datetime.datetime.now()),
                }
            )

    def is_market_open(self):
        # 5=Sabtu, 6=Minggu -> Tutup
        return datetime.datetime.now().weekday() < 5

    async def run(self):
        await self.init()
        logger.info("🧠 Brain Running...")

        while True:
            # --- TIMEOUT GUARD (Anti Deadlock) ---

            if not self.is_market_open():
                logger.info("💤 Weekend - Market Closed. Sleeping...")
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
                    logger.warning("⚠️ Executor Timeout! Resetting State.")
                    self.is_pending_order = False
                    self.pending_start_time = None

            candles = await streamor.consume_market_data()
            if not candles:
                await asyncio.sleep(0.1)
                continue

            for c in candles:
                self.buffer.append(c)
                await self.manage_positions(c)

                if await state_manager.get_active_position() or self.is_pending_order:
                    continue
                if len(self.buffer) < settings.SEQ_LEN:
                    continue

                # LOGIKA MODEL DISINI (Sementara di-pass)
                # Jika ingin tes order otomatis, ubah "HOLD" jadi "BUY" manual
                action = "HOLD"

                if action == "BUY":
                    logger.info("🚀 Signal BUY Sent")
                    self.is_pending_order = True
                    self.pending_start_time = datetime.datetime.now()
                    await streamor.push_signal(
                        {
                            "action": "BUY",
                            "symbol": settings.ACTIVE_SYMBOL,
                            "entry_price": c["close"],
                            "timestamp": str(datetime.datetime.now()),
                        }
                    )


if __name__ == "__main__":
    asyncio.run(Brain().run())
    asyncio.run(Brain().run())
