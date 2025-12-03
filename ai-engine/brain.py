import asyncio
import datetime
import logging
from collections import deque

import pandas as pd
import torch
from config import settings
from database import TradeLog, get_session_local
from features import fetcher, processor
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain..")


class PositionManager:
    def __init__(self):
        self.db = get_session_local()()

    def open(self, signal):
        state_manager.set_active_position(
            signal["symbol"],
            signal["action"],
            signal["price"],
            settings.LOT_SIZE,
            signal["tp"],
            signal["sl"],
        )
        trade = TradeLog(
            symbol=signal["symbol"],
            action=signal["action"],
            entry_price=signal["price"],
            tp_price=signal["tp"],
            sl_price=signal["sl"],
            status="OPEN",
        )
        self.db.add(trade)
        self.db.commit()
        streamor.push_signal(signal)
        logger.info(
            f"📈 infoo pebtibg OPEN: {signal['action']} {signal['symbol']} {signal['price']}"
        )

    def check_exit(self, price, time):
        pos = state_manager.get_active_position()
        if not pos:
            return

        entry, side = float(pos["entry_price"]), pos["side"]
        tp, sl = float(pos["tp"]), float(pos["sl"])
        reason = None

        if side == "BUY":
            if price >= tp:
                reason = "TP_HIT"
            elif price <= sl:
                reason = "SL_HIT"
        elif side == "SELL":
            if price <= tp:
                reason = "TP_HIT"
            elif price >= sl:
                reason = "SL_HIT"

        if reason:
            self.close(pos, price, reason, time)

    def close(self, pos, price, reason, time):
        state_manager.clear_active_position()
        trade = (
            self.db.query(TradeLog)
            .filter(TradeLog.status == "OPEN")
            .order_by(TradeLog.timestamp.desc())
            .first()
        )
        pnl = 0
        if trade:
            trade.exit_price = price
            trade.exit_time = datetime.datetime.now()
            trade.status = reason
            diff = price - trade.entry_price
            if trade.action == "SELL":
                diff = -diff
            trade.pnl = diff * settings.CONTRACT_SIZE * settings.LOT_SIZE
            pnl = trade.pnl
            self.db.commit()

        streamor.push_signal(
            {
                "type": "CLOSE",
                "symbol": pos["symbol"],
                "action": reason,
                "price": price,
                "pnl": pnl,
                "timestamp": str(time),
            }
        )
        logger.info(f"💰 CLOSED: {reason} PnL: {pnl}")


class Brain:
    def __init__(self):
        self.buffer = deque(maxlen=100)
        self.pos_mgr = PositionManager()
        self.model = None

    async def init(self):
        if not processor.load_scaler():
            raise Exception("No Scaler")
        self.model = TimeSeriesTransformer(input_dim=4)
        self.model.load_state_dict(torch.load(settings.MODEL_FILE))
        self.model.eval()

        # Warmup
        df = await fetcher.fetch_market_data(settings.SYMBOL_YF, period="5d")
        for i, row in df.tail(100).iterrows():
            self.update_buffer(row)
            logger.info(
                f"📊 Warmup: {i} o={row['open']} h={row['high']} l={row['low']} c={row['close']} v={row['volume']}"
            )

    def update_buffer(self, data):
        try:
            self.buffer.append(
                {
                    "open": float(data["open"]),
                    "high": float(data["high"]),
                    "low": float(data["low"]),
                    "close": float(data["close"]),
                    "volume": float(data["volume"]),
                }
            )
            logger.info(f"📊 Buffer length: {len(self.buffer)}")
            logger.info(f"📊 Buffer updated: {data['close']}")
            self.buffer = list(dict.fromkeys(self.buffer))
        except:
            logger.error("📉 Buffer update failed")
            pass

    async def run(self):
        await self.init()
        logger.info("🧠 Brain Running...")
        while True:

            candles = streamor.consume_market_data("brain_1")
            if not candles:
                await asyncio.sleep(0.1)
                continue

            for c in candles:
                price = c["close"]
                self.pos_mgr.check_exit(price, c["timestamp"])
                self.update_buffer(c)

                if len(self.buffer) < 50 or state_manager.get_active_position():
                    continue

                df = pd.DataFrame(self.buffer)
                _, feats = processor.process(df)
                if len(feats) < settings.SEQ_LEN:
                    continue

                tensor = torch.FloatTensor(feats[-settings.SEQ_LEN :]).unsqueeze(0)
                with torch.no_grad():
                    probs = self.model(tensor).numpy()[0]

                action = "HOLD"
                if probs[0] > 0.7:
                    action = "BUY"
                elif probs[1] > 0.7:
                    action = "SELL"

                if action != "HOLD":
                    tp = (
                        price * (1 + settings.TAKE_PROFIT_PCT)
                        if action == "BUY"
                        else price * (1 - settings.TAKE_PROFIT_PCT)
                    )
                    sl = (
                        price * (1 - settings.STOP_LOSS_PCT)
                        if action == "BUY"
                        else price * (1 + settings.STOP_LOSS_PCT)
                    )
                    self.pos_mgr.open(
                        {
                            "symbol": settings.SYMBOL_BINANCE,
                            "action": action,
                            "price": price,
                            "tp": tp,
                            "sl": sl,
                            "confidence": float(max(probs)),
                            "timestamp": c["timestamp"],
                        }
                    )
                    logger.info(f"📈 OPEN: {action} {settings.SYMBOL_BINANCE} {price}")
                    logger.info(f"📊 TP: {tp} SL: {sl}")
                    logger.info(f"📅 Timestamp: {c['timestamp']}")


if __name__ == "__main__":
    asyncio.run(Brain().run())
