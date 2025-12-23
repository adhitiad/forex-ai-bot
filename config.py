import os

import MetaTrader5 as mt5
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- IDENTITAS ---
    VERSION: str = "3.0.0-M15"
    APP_NAME: str = "Forex AI Bot (Multi-Pair Enterprise)"

    # --- MT5 CONFIG ---
    MT5_LOGIN: int = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER: str = os.getenv("MT5_SERVER", "FBS-Real")
    MT5_PATH: str = os.getenv("MT5_PATH", r"C:\Program Files\FBS MT5\terminal64.exe")

    # --- MULTI PAIR SETUP ---
    # Masukkan semua pair yang mau ditradingkan di sini
    FOREX_SYMBOLS: list = ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"]
    STOCK_SYMBOLS: list = ["BBCA.JK", "TLKM.JK", "BBRI.JK", "GOTO.JK"]

    TIMEFRAME: int = mt5.TIMEFRAME_M15
    LEVERAGE: int = 500

    # Risk Management
    RISK_PER_TRADE: float = 0.01
    MAX_OPEN_TRADES: int = 3

    # --- AI PARAMETERS ---
    SEQ_LEN: int = 60
    PREDICTION_WINDOW: int = 4

    # --- PATHS (DINAMIS) ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    def get_model_path(self, symbol):
        """Mendapatkan path model khusus untuk pair tertentu"""
        return os.path.join(self.BASE_DIR, "data", f"model_{symbol}.pth")

    def get_scaler_path(self, symbol):
        """Mendapatkan path scaler khusus untuk pair tertentu"""
        return os.path.join(self.BASE_DIR, "data", f"scaler_{symbol}.pkl")

    # --- TRADING CONFIG (INTRADAY M15) ---
    # Gunakan M15 untuk Intraday (Menangkap pergerakan harian)

    # Lot Size (Jika 0, akan dihitung otomatis berdasarkan Risk & Leverage)
    # Jika saldo kecil (<$100), set manual misal 0.01
    TRADE_LOTS: float = 0.0

    # Take Profit & Stop Loss (Estimasi Awal, nanti diatur dynamic oleh Brain)
    # Untuk M15: SL 20 pips, TP 40 pips (Ratio 1:2)
    SL_PIPS: int = 20
    TP_PIPS: int = 40

    # --- AI HYPERPARAMETERS (Setting Otak) ---
    # Ambang Batas Volatilitas (Lebih kecil dari H1 karena M15 gerakan lebih halus)
    # 0.0010 = 10 Pips change. Jika prediksi perubahan < 10 pips, anggap HOLD.
    VOLATILITY_THRESHOLD: float = 0.0010

    # Training
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-5
    EPOCHS: int = 50
    RETRAIN_ON_LOSS_COUNT: int = 3
    AUTO_RETRAIN: bool = True

    # --- PATHS ---
    MODEL_FILE: str = os.path.join(
        BASE_DIR, "data", "forex_ai_model_m15.pth"
    )  # Nama file beda biar gak ketukar
    SCALER_FILE: str = os.path.join(BASE_DIR, "data", "forex_ai_scaler_m15.pkl")

    # --- REDIS ---
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # Channels
    CHANNEL_MARKET: str = "market_data"
    CHANNEL_AI_ANALYSIS: str = "forex_ai_analysis"
    CHANNEL_CONFIRMATION: str = "trade_confirmation"
    CHANNEL_SYSTEM: str = "system_events"

    # --- NOTIFIKASI ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- CLOUD SYNC ---
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")
    CLOUD_MODEL_NAME: str = "forex_ai_model_m15"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields from .env file


settings = Settings()
