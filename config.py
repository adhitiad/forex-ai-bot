import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- BASIC ---
    ACTIVE_SYMBOL: str = "EURUSD=X"
    YFINANCE_SYMBOL: str = "EURUSD=X"
    TIMEFRAME: str = "1h"
    SEQ_LEN: int = 30
    TRADE_UNITS: int = 1000
    ASSET_TYPE: str = "FOREX"

    # --- INFRA ---
    REDIS_HOST: str = "redis"  # Gunakan "localhost" jika tanpa docker
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # Menggunakan connection string yang Anda berikan
    DATABASE_URL: str = ""
    # --- GRPC SERVER CONFIG ---
    GRPC_SERVER_HOST: str = "[::]:50051"  # Port standar gRPC

    # --- CHANNELS ---
    CHANNEL_MARKET: str = "channel_market"
    CHANNEL_SIGNALS: str = "channel_signals"  # Output Brain V1
    CHANNEL_CONFIRMATION: str = "channel_confirmation"
    CHANNEL_AI_ANALYSIS: str = "channel_ai_analysis"  # Output Brain V2
    CHANNEL_SYSTEM: str = "channel_system"  # Status Training/Safety

    # --- AI BRAIN V2 ---
    GROQ_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "forex-memory"
    NVIDIA_API_KEY: str = ""

    # --- CLOUDINARY ---
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUD_MODEL_NAME: str = "forex_ai_model_latest"

    # --- SELF-HEALING ---
    AUTO_RETRAIN: bool = True
    RETRAIN_ON_LOSS_COUNT: int = 3

    # --- SAFETY & RISK ---
    MAX_DAILY_LOSS_PERCENT: float = 2.0
    USE_ATR_FOR_SL: bool = True
    ATR_PERIOD: int = 14

    # --- NOTIFICATION ---
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # --- OANDA ---
    OANDA_ACCESS_TOKEN: str = ""
    OANDA_ENV: str = "practice"
    OANDA_SYMBOL: str = "EUR_USD"
    OANDA_ACCOUNT_ID: str = ""

    @property
    def MODEL_FILE(self) -> str:
        return f"data/forex_model.pth"

    @property
    def SCALER_FILE(self) -> str:
        return f"data/forex_scaler.pkl"

    class Config:
        extra = "ignore"
        env_file = ".env"


settings: Settings = Settings()

if not os.path.exists("data"):
    os.makedirs("data")
