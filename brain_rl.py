import asyncio
import datetime
import logging
from collections import deque

import pandas as pd
from stable_baselines3 import PPO

from config import settings
from features import processor
from logging_config import setup_logger
from stream_manager import streamor

logger = setup_logger("Brain-RL")


class BrainRL:
    def __init__(self):
        self.buffers = {sym: deque(maxlen=60) for sym in settings.ACTIVE_SYMBOLS}
        self.model = None

    def load_model(self):
        try:
            self.model = PPO.load("data/rl_model_ppo")
            logger.info("🧠 RL Agent Loaded.")
        except:
            logger.warning("⚠️ RL Model not found.")

    async def run(self):
        self.load_model()
        streamor.connect()
        logger.info("🚀 Brain RL Running...")
        while True:
            candles = await streamor.consume_market_data(
                group="rl_grp", consumer="rl_agent"
            )
            if not candles:
                await asyncio.sleep(0.1)
                continue

            for c in candles:
                sym = c.get("symbol")
                if sym not in self.buffers:
                    continue
                self.buffers[sym].append(c)

                if len(self.buffers[sym]) >= 30 and self.model:
                    df = pd.DataFrame(list(self.buffers[sym]))
                    _, features = processor.process(df)
                    if len(features) < 30:
                        continue

                    action, _ = self.model.predict(features[-30:], deterministic=True)
                    act_str = (
                        "BUY" if action == 1 else "SELL" if action == 2 else "HOLD"
                    )

                    if act_str != "HOLD":
                        await streamor.push_signal(
                            {
                                "source": "BRAIN_RL",
                                "symbol": sym,
                                "action": act_str,
                                "confidence": 0.85,
                                "price": c["close"],
                                "timestamp": str(datetime.datetime.now()),
                            }
                        )


if __name__ == "__main__":
    asyncio.run(BrainRL().run())
