import asyncio
import json
import logging
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import internal
from config import settings
from features import processor
from ingestor import DataFetcher  # Use DataFetcher for data fetching

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RL-Trainer-Multi")


# --- ENVIRONMENT TRADING (Sama seperti sebelumnya) ---
class ForexTradingEnv(gym.Env):
    def __init__(self, df, features):
        super(ForexTradingEnv, self).__init__()
        self.df = df
        self.features = features
        self.current_step = 0
        self.max_steps = len(features) - 1
        self.action_space = spaces.Discrete(3)  # 0:Hold, 1:Buy, 2:Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.features[self.current_step], {}

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        if terminated:
            return self.features[self.current_step - 1], 0, True, False, {}

        next_price = self.df.iloc[self.current_step]["close"]
        price_change = (next_price - current_price) / current_price

        reward = 0.0
        if action == 1:  # BUY
            reward = price_change
        elif action == 2:  # SELL
            reward = -price_change
        elif action == 0:  # HOLD
            reward = -0.00001  # Kecil penalti biar gak pasif

        return self.features[self.current_step], reward, terminated, truncated, {}


# --- FUNGSI TRAIN PER SYMBOL ---
async def train_single_symbol(symbol):
    logger.info(f"🚀 Training Brain-RL (PPO) for {symbol}...")

    # 1. Fetch Data Spesifik Symbol untuk RL training
    fetcher = DataFetcher([symbol])  # Initialize with single symbol

    # Use the historical data fetching method for training
    df = await fetcher.fetch_market_data(symbol=symbol, days=730)

    if df.empty:
        logger.error(f"❌ Data kosong untuk {symbol}, skip RL training.")
        return

    # 2. Process Features (Penting: Scaler per Symbol)
    # is_training=True agar scaler di-fit ulang khusus symbol ini
    df, scaled_features = processor.process(df, symbol=symbol, is_training=True)

    if len(scaled_features) < 100:
        logger.error(f"❌ Data terlalu sedikit untuk {symbol}.")
        return

    # 3. Setup Env & Model
    env = DummyVecEnv([lambda: ForexTradingEnv(df, scaled_features)])
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=2048)

    # 4. Training
    logger.info(f"🧠 Learning process started for {symbol}...")
    model.learn(total_timesteps=50000)  # Bisa dinaikkan

    # 5. Save Model dengan Nama Unik
    # Contoh: data/rl_model_EURUSD
    save_name = f"rl_model_{symbol}"
    save_path = os.path.join(settings.BASE_DIR, "data", save_name)

    model.save(save_path)
    logger.info(f"✅ Brain-RL Saved: {save_path}.zip")


# --- MAIN LOOP ---
async def main():
    logger.info(f"Starting Multi-Symbol RL Training for: {settings.ACTIVE_SYMBOLS}")

    for symbol in settings.ACTIVE_SYMBOLS:
        await train_single_symbol(symbol)

    logger.info("🎉 All RL Agents Trained Successfully.")

    # Notifikasi System
    try:
        import redis

        r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB
        )
        r.publish(
            settings.CHANNEL_SYSTEM, json.dumps({"event": "RL_TRAINING_COMPLETED"})
        )
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())
