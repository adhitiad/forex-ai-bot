import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- YFINANCE PAPER MODE ---
    ASSET_TYPE: str = "FOREX"

    # Gunakan simbol Yahoo Finance
    # Contoh: "EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "XAUUSD=X" (Gold)
    ACTIVE_SYMBOL: str = "EURUSD=X"

    # Simbol Training (Sama dengan Active)
    YFINANCE_SYMBOL: str = "EURUSD=X"

    # Timeframe (Format Yahoo: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    # Gunakan "5m" atau "15m" untuk scalping bot
    TIMEFRAME: str = "1h"
    SEQ_LEN: int = 30

    # Risk Management (Simulasi)
    TRADE_UNITS: int = 1000
    TAKE_PROFIT_PIPS: int = 20
    STOP_LOSS_PIPS: int = 15

    # Infrastructure
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    DATABASE_URL: str = "sqlite:///./trading_paper.db"

    # Redis Channels
    CHANNEL_MARKET: str = "channel_market"
    CHANNEL_SIGNALS: str = "channel_signals"
    CHANNEL_CONFIRMATION: str = "channel_confirmation"

    @property
    def MODEL_FILE(self) -> str:
        return f"data/forex_model.pth"

    @property
    def SCALER_FILE(self) -> str:
        return f"data/forex_scaler.pkl"

    class Config:
        extra = "ignore"
        env_file = ".env"


settings = Settings()
