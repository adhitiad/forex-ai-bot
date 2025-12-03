from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Trading
    SYMBOL_YF: str = "EURUSD=X"
    SYMBOL_BINANCE: str = "XAUUSD"
    TIMEFRAME: str = "1h"
    SEQ_LEN: int = 30
    PREDICT_WINDOW: int = 4

    # Risk Management
    LOT_SIZE: float = 0.01
    TAKE_PROFIT_PCT: float = 0.005  # 0.5%
    STOP_LOSS_PCT: float = 0.005  # 0.5%
    CONTRACT_SIZE: int = 100

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str

    # Database
    DATABASE_URL: str

    # Paths
    MODEL_FILE: str = "trained_model.pth"
    SCALER_FILE: str = "scaler.pkl"

    class Config:
        """
        Pydantic settings configuration.
        """

        env_file = ".env"


settings = Settings()  # type: ignore
