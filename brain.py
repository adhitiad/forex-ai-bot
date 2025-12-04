import asyncio
import datetime
import json
import logging
from collections import deque

import pandas as pd
import torch

from config import settings
from database import TradeLog, get_db
from features import processor
from model import TimeSeriesTransformer
from state_manager import state_manager
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Brain")


class Brain:
    def __init__(self):
        self.buffer = deque(maxlen=100)
        self.model = None
        self.latest_sentiment = 0.0
        self.db_gen = get_db()
        self.db = next(self.db_gen)

    async def init(self):
        if not processor.load_scaler():
            return
        self.model = TimeSeriesTransformer(input_dim=4)
        try:
            self.model.load_state_dict(torch.load(settings.MODEL_FILE))
            self.model.eval()
            logger.info("🧠 Model Loaded")
        except:
            pass
        asyncio.create_task(self.listen_sentiment())

    async def listen_sentiment(self):
        if not streamor.r:
            return
        pubsub = streamor.r.pubsub()
        await pubsub.subscribe("channel_sentiment")
        async for m in pubsub.listen():
            if m["type"] == "message":
                self.latest_sentiment = float(json.loads(m["data"])["sentiment_score"])

    async def run(self):
        await self.init()
        while True:
            if streamor and streamor.r:
                candles = await streamor.consume_market_data()
            else:
                candles = []
            if not candles:
                await asyncio.sleep(0.1)
                continue

            for c in candles:
                # 1. Update Buffer
                self.buffer.append(c)
                if len(self.buffer) < settings.SEQ_LEN:
                    continue

                # 2. Process
                df = pd.DataFrame(self.buffer)
                _, feats = processor.process(df)
                if len(feats) < settings.SEQ_LEN:
                    continue

                if not self.model:
                    continue

                # 3. Predict
                tensor = torch.FloatTensor(feats[-settings.SEQ_LEN :]).unsqueeze(0)
                with torch.no_grad():
                    probs = self.model(tensor).numpy()[0]  # [Hold, Buy, Sell]

                # 4. Fusion Logic (AI + Sentiment)
                buy_conf = probs[1] + (0.1 if self.latest_sentiment > 0.5 else 0)
                sell_conf = probs[2] + (0.1 if self.latest_sentiment < -0.5 else 0)

                # 5. Execute
                action = "HOLD"
                if buy_conf > 0.7:
                    action = "BUY"
                elif sell_conf > 0.7:
                    action = "SELL"

                if action != "HOLD" and not state_manager.get_active_position():
                    # Simplified entry logic...
                    tp = c["close"] * (1.01 if action == "BUY" else 0.99)
                    sl = c["close"] * (0.99 if action == "BUY" else 1.01)

                    trade = TradeLog(
                        symbol=c.get("source", "UNK"),
                        action=action,
                        entry_price=c["close"],
                        tp_price=tp,
                        sl_price=sl,
                        status="OPEN",
                    )
                    self.db.add(trade)
                    self.db.commit()

                    state_manager.set_active_position(
                        c.get("source"), action, c["close"], 0.01, tp, sl
                    )
                    if streamor:
                        await streamor.push_signal(
                            {
                                "action": action,
                                "entry_price": c["close"],
                                "symbol": settings.ACTIVE_SYMBOL,
                                "status": "OPEN",
                                "timestamp": str(datetime.datetime.now()),
                            }
                        )
                    logger.info(f"🚀 {action} Signal")


if __name__ == "__main__":
    asyncio.run(Brain().run())
