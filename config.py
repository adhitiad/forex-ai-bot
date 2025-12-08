import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- CRYPTO MODE ---
    ASSET_TYPE: str = "CRYPTO"

    # Data Source Selection
    USE_YFINANCE: bool = True  # Set True jika exchange diblokir, False untuk CCXT

    # Alternatif exchange yang tidak diblokir di Indonesia (jika USE_YFINANCE=False):
    # - kraken: Stabil, tidak perlu VPN
    # - bybit: Exchange Asia
    # - kucoin: Alternatif populer
    EXCHANGE_ID: str = "bybit"  # Gunakan bybit atau kraken
    ACTIVE_SYMBOL: str = "BTC/USDT"
    YFINANCE_SYMBOL: str = "BTC-USD"  # Symbol untuk Yahoo Finance

    # API & Data
    TIMEFRAME: str = "1h"
    SEQ_LEN: int = 30

    # Credentials (Wajib diisi di .env)
    TOKOCRYPTO_API_KEY: str = ""
    TOKOCRYPTO_SECRET: str = ""
    API_SECRET: str = ""

    # Risk Management
    TAKE_PROFIT_PCT: float = 0.081  # Target 8,1%
    STOP_LOSS_PCT: float = 0.01  # Stop Loss 1%

    # Infrastructure
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    DATABASE_URL: str = "sqlite:///./trading.db"

    # Redis Channels (V2 Architecture)
    CHANNEL_MARKET: str = "channel_market"
    CHANNEL_SIGNALS: str = "channel_signals"
    CHANNEL_SENTIMENT: str = "channel_sentiment"
    CHANNEL_CONFIRMATION: str = "channel_confirmation"  # Channel Baru

    # Model Paths
    @property
    def MODEL_FILE(self) -> str:
        return f"data/crypto_model_{self.ASSET_TYPE.lower()}.pth"

    @property
    def SCALER_FILE(self) -> str:
        return f"data/crypto_scaler_{self.ASSET_TYPE.lower()}.pkl"

    class Config:
        extra = "ignore"
        env_file = ".env"


settings = Settings()
