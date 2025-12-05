import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

os.makedirs("data", exist_ok=True)


class Settings(BaseSettings):
    # --- FOREX MODE ON ---
    ASSET_TYPE: str = "FOREX"
    ACTIVE_SYMBOL: str = "EURUSD=X"  # Default pair (Euro vs US Dollar)

    # API & Data
    TIMEFRAME: str = "1h"
    SEQ_LEN: int = 30
    PREDICT_WINDOW: int = 4

    # Isi API Key NVIDIA & Database di .env
    NVIDIA_API_KEY: str = ""
    DATABASE_URL: str = (
        "sqlite:///./trading.db"  # Default SQLite (Bisa ganti Postgres di .env)
    )
    API_SECRET: str = ""

    # --- RISK MANAGEMENT KHUSUS FOREX ---
    # Forex bergerak dalam pecahan kecil (Pips).
    # 0.002 = 0.2% = Sekitar 20 Pips (Target realistis H1)
    # 0.001 = 0.1% = Sekitar 10 Pips (Stop Loss ketat)
    TAKE_PROFIT_PCT: float = 0.003  # Target 30 Pips
    STOP_LOSS_PCT: float = 0.0015  # Stop Loss 15 Pips

    LOT_SIZE: float = 0.1  # 0.1 Lot (Mini Lot)
    LEVERAGE: int = 100  # Estimasi leverage (hanya untuk log)

    # Infrastructure
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # Paths
    MODEL_FILE: str = (
        "data/forex_model.pth"  # Nama file beda biar gak ketukar sama crypto
    )
    SCALER_FILE: str = "data/forex_scaler.pkl"

    class Config:
        extra = "ignore"
        env_file = ".env"


settings = Settings()
