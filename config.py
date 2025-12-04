import os

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

os.makedirs("data", exist_ok=True)


class Settings(BaseSettings):
    ASSET_TYPE: str = "CRYPTO"

    # Symbols
    SYMBOL_FOREX: str = "EURUSD=X"
    SYMBOL_CRYPTO: str = "BTC/USDT"

    # Exchange
    EXCHANGE_ID: str = "binance"
    API_KEY: str = ""
    API_SECRET: str = ""

    # AI & Data
    TIMEFRAME: str = "1h"
    SEQ_LEN: int = 30
    PREDICT_WINDOW: int = 4
    GROQ_API_KEY: str = ""

    # Risk Management
    LOT_SIZE: float = 0.01
    TAKE_PROFIT_PCT: float = 0.01
    STOP_LOSS_PCT: float = 0.005
    CONTRACT_SIZE: int = 100

    # Infrastructure
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    DATABASE_URL: str = "sqlite:///./trading.db"

    # Paths
    MODEL_FILE: str = "data/trained_model.pth"
    SCALER_FILE: str = "data/scaler.pkl"

    class Config:
        extra = "ignore"
        env_file = ".env"

    @property
    def ACTIVE_SYMBOL(self):
        return self.SYMBOL_CRYPTO if self.ASSET_TYPE == "CRYPTO" else self.SYMBOL_FOREX


settings = Settings()
