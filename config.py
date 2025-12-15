import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- ASSETS ---
    ACTIVE_SYMBOLS: list = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "XAUUSD=X"]

    # --- INFRA ---
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    DATABASE_URL: str = ""

    # --- CHANNELS ---
    CHANNEL_MARKET: str = "channel_market"
    CHANNEL_SIGNALS: str = "channel_signals"
    CHANNEL_AI_ANALYSIS: str = "channel_ai_analysis"
    CHANNEL_CONFIRMATION: str = "channel_confirmation"
    CHANNEL_SYSTEM: str = "channel_system"

    # --- API KEYS ---
    OANDA_ACCESS_TOKEN: str = os.getenv("OANDA_ACCESS_TOKEN", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_ENV: str = os.getenv("OANDA_ENV", "practice")

    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    HF_MODEL_ID: str = "zai-org/GLM-4.6"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_ID: str = "llama3-70b-8192"

    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- RISK & LOGIC ---
    MAX_OPEN_POSITIONS: int = 3
    MAX_DAILY_LOSS_PERCENT: float = 3.0
    VOLATILITY_THRESHOLD: float = 0.0025
    EXECUTION_MODE: str = "SMART"  # SMART or STANDARD
    LIMIT_CHASE_TIMEOUT: int = 20

    SEQ_LEN: int = 60  # Panjang Sequence untuk Model
    PREDICTION_WINDOW: int = 15  # Jendela Prediksi (15 candle ke depan)
    STRIDE: int = 1  # Langkah untuk Sliding Window
    BATCH_SIZE: int = 64  # Ukuran Batch untuk Training
    LEARNING_RATE: float = 1e-4  # Learning Rate untuk Optimizer
    WEIGHT_DECAY: float = 1e-5  # Weight Decay untuk Optimizer
    GRADIENT_CLIP: float = 1.0  # Gradient Clipping untuk Stabilitas


settings = Settings()
