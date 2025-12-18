import asyncio
import logging
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import modul internal
from config import settings
from features import fetcher, processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RL-Trainer")


# --- 1. MEMBUAT ENVIRONMENT TRADING ---
class ForexTradingEnv(gym.Env):
    def __init__(self, df, features):
        super(ForexTradingEnv, self).__init__()
        self.df = df
        self.features = features  # Data ternormalisasi (Input AI)
        self.current_step = 0
        self.max_steps = len(features) - 1

        # Aksi: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observasi: 4 Fitur (Returns, RSI, EMA, ATR)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # State Trading
        self.position = 0  # 0=None, 1=Buy, -1=Sell
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self.features[self.current_step], {}

    def step(self, action):
        # Data saat ini
        current_price = self.df.iloc[self.current_step]["close"]

        # Bergerak ke candle berikutnya
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        next_price = self.df.iloc[self.current_step]["close"]
        reward = 0.0

        # --- LOGIKA REWARD (INTI KECERDASAN) ---
        # Kita beri reward jika aset tumbuh, penalty jika rugi

        # Hitung perubahan harga (%)
        price_change = (next_price - current_price) / current_price

        if action == 1:  # BUY
            # Untung jika harga naik, rugi jika turun. Kurangi spread/fee (estimasi 0.0001)
            reward = price_change - 0.0001
        elif action == 2:  # SELL
            # Untung jika harga turun
            reward = -price_change - 0.0001
        elif action == 0:  # HOLD
            # Netral, tapi beri sedikit penalty agar tidak pasif selamanya
            reward = -0.00001

        # Return: Obs Baru, Reward, Selesai?, Info
        return self.features[self.current_step], reward, terminated, truncated, {}


# --- 2. FUNGSI TRAINING UTAMA ---
async def train_rl_agent():
    logger.info(
        f"🚀 Memulai Training Brain-RL (PPO) untuk {settings.ACTIVE_SYMBOLS[0]}..."
    )

    # A. Ambil Data
    logger.info("📥 Fetching Market Data...")
    df = await fetcher.fetch_market_data(days=730)  # 2 Tahun data
    if df.empty:
        logger.error("❌ Data kosong!")
        return

    # B. Proses Fitur (Indikator & Scaling)
    # Gunakan processor yang sama dengan sistem live agar cerdasnya sinkron
    df, scaled_features = processor.process(df, is_training=True)

    logger.info(f"📊 Data Siap: {len(scaled_features)} candles.")

    # C. Setup Environment
    # Kita bungkus env agar kompatibel dengan Stable Baselines3
    env = DummyVecEnv([lambda: ForexTradingEnv(df, scaled_features)])

    # D. Setup Model PPO
    # MlpPolicy = Multi Layer Perceptron (Otak standar untuk data angka)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,  # Default bagus untuk PPO
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # E. Proses Belajar (Training)
    # total_timesteps semakin besar = semakin pintar (tapi lama)
    # 100.000 langkah cukup untuk permulaan agar paham tren dasar
    logger.info("🧠 Sedang melatih agen... (Ini memakan waktu)")
    model.learn(total_timesteps=100000)

    # F. Simpan Otak
    save_path = "data/rl_model_ppo"
    model.save(save_path)
    logger.info(f"✅ Brain-RL Selesai Dilatih! Disimpan ke: {save_path}.zip")

    # Notifikasi Redis (opsional, biar keren di log runner)
    try:
        import redis

        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        r.publish(settings.CHANNEL_SYSTEM, json.dumps({"event": "RL_TRAINING_DONE"}))
    except:
        pass


if __name__ == "__main__":
    asyncio.run(train_rl_agent())
