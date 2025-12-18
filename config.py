# File: config.py
import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- ASSETS ---
    # Sesuaikan simbol dengan nama di FBS (misal: "EURUSD" atau "EURUSD.m")
    ACTIVE_SYMBOLS: list = ["EURUSD", "GBPUSD", "XAUUSD"]
    YFINANCE_SYMBOL: str = "EURUSD=X"
    TIMEFRAME: str = "1h"

    # --- INFRA ---
    REDIS_HOST: str = (
        "localhost"  # Ganti 'redis' jika jalan di local windows tanpa docker network
    )
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    DATABASE_URL: str = ""

    # --- CHANNELS ---
    CHANNEL_MARKET: str = "channel_market"
    CHANNEL_SIGNALS: str = "channel_signals"
    CHANNEL_AI_ANALYSIS: str = "channel_ai_analysis"
    CHANNEL_CONFIRMATION: str = "channel_confirmation"
    CHANNEL_SYSTEM: str = "channel_system"

    # --- MT5 CONFIG (Ganti OANDA) ---
    MT5_LOGIN: int = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER: str = os.getenv("MT5_SERVER", "FBS-Demo")  # Cek di File > Open Account
    MT5_PATH: str = os.getenv("MT5_PATH", "C:\\Program Files\\FBS MT5\\terminal64.exe")

    # --- API KEYS LAIN ---
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    HF_MODEL_ID: str = "zai-org/GLM-4.6"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_ID: str = "llama-3.3-70b-versatile"
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- PINECONE ---
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "")

    # --- CLOUD & FILES ---
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")

    MODEL_FILE: str = "data/model.pth"
    CLOUD_MODEL_NAME: str = "forex_ai_model.pth"
    SCALER_FILE: str = "data/scaler.pkl"

    # --- RISK & LOGIC ---
    MAX_OPEN_POSITIONS: int = 3
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    VOLATILITY_THRESHOLD: float = 0.0050
    EXECUTION_MODE: str = "SMART"
    LIMIT_CHASE_TIMEOUT: int = 20
    RETRAIN_ON_LOSS_COUNT: int = 5
    AUTO_RETRAIN: bool = True

    # Model Params
    SEQ_LEN: int = 60
    PREDICTION_WINDOW: int = 15
    STRIDE: int = 1
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    GRADIENT_CLIP: float = 1.0

    # Exec Params
    TRADE_LOTS: float = 0.01  # Default Lot


settings = Settings()
