import asyncio
import logging
from collections import deque

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import settings
from features import processor
from logging_config import setup_logger
from stream_manager import streamor

logger = setup_logger("Brain-RL")


class BrainRL:
    def __init__(self):
        self.buffers = {sym: deque(maxlen=100) for sym in settings.ACTIVE_SYMBOLS}
        self.model = None
        self.model_path = "data/rl_model_ppo"  # Pastikan file ini ada hasil training

    def load_model(self):
        try:
            # Menggunakan device cpu agar tidak berebut GPU dengan Brain V1
            self.model = PPO.load(self.model_path, device="cpu")
            logger.info("🧠 RL Agent Loaded (PPO).")
        except:
            logger.warning(
                f"⚠️ RL Model not found at {self.model_path}. RL will stay silent."
            )

    def warmup_data(self):
        """Warmup penting untuk indikator"""
        if not mt5.initialize():
            return
        logger.info("🔥 Warming up RL buffers...")
        for sym in settings.ACTIVE_SYMBOLS:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 100)
            if rates is not None:
                for r in rates:
                    data = {
                        "symbol": sym,
                        "close": float(r["close"]),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "volume": float(r["tick_volume"]),
                        "timestamp": str(r["time"]),
                    }
                    self.buffers[sym].append(data)

    async def run(self):
        self.load_model()
        self.warmup_data()
        await streamor.connect()

        logger.info("🚀 Brain RL (Expert) Running...")
        while True:
            # Gunakan consumer group agar tidak berebut pesan dengan Brain V1
            candles = await streamor.consume_market_data(
                group="rl_group", consumer="rl_node_1"
            )

            if not candles:
                await asyncio.sleep(0.01)
                continue

            for c in candles:
                sym = c.get("symbol")
                if sym not in self.buffers:
                    continue
                self.buffers[sym].append(c)

                # Syarat Minimal Data
                if len(self.buffers[sym]) >= 60 and self.model:
                    try:
                        df = pd.DataFrame(list(self.buffers[sym]))
                        # Gunakan processor fitur yang sama agar data konsisten
                        _, features = processor.process(df)

                        if len(features) < 1:
                            continue

                        # === INFERENCE ===
                        # Obs terakhir untuk prediksi langkah selanjutnya
                        obs = features[-1:]
                        action, _ = self.model.predict(obs, deterministic=True)

                        # Mapping Action Space: 0=Hold, 1=Buy, 2=Sell (Contoh Discrete)
                        act_str = "HOLD"
                        if action == 1:
                            act_str = "BUY"
                        elif action == 2:
                            act_str = "SELL"

                        if act_str != "HOLD":
                            # Kirim sinyal ke Fusion Engine
                            payload = {
                                "source": "BRAIN_RL",
                                "symbol": sym,
                                "action": act_str,
                                "price": c["close"],
                                "confidence": 0.8,  # RL Model biasanya confident
                            }
                            await streamor.push_signal(payload)

                    except Exception as e:
                        logger.error(f"RL Inference Error: {e}")


if __name__ == "__main__":
    asyncio.run(BrainRL().run())
