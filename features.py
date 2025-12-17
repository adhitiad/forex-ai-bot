import logging
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

from config import settings

logger = logging.getLogger("Features")


class FeatureEngineer:
    def __init__(self, path=settings.SCALER_FILE):
        self.path = path
        self.scaler = RobustScaler()
        # Kolom yang akan masuk ke Model AI
        self.feature_cols = ["returns", "RSI_14", "EMA_20", "ATRr_14"]

    def add_indicators(self, df):
        """Hanya menambahkan indikator teknikal tanpa scaling"""
        if df.empty:
            return df

        df = df.copy()

        # 1. Returns
        df["returns"] = df["close"].pct_change()

        # 2. RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # 3. EMA 20
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

        # 4. ATR 14
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATRr_14"] = tr.rolling(window=14).mean()

        # Drop NaN akibat rolling window
        df.dropna(inplace=True)
        return df

    def get_scaler(self):
        """Load scaler jika ada, atau return None"""
        if os.path.exists(self.path):
            try:
                return joblib.load(self.path)
            except:
                return None
        return None

    def fit_scaler(self, train_df):
        """Fit scaler HANYA pada data training"""
        if train_df.empty:
            return

        raw_train = train_df[self.feature_cols].values
        self.scaler.fit(raw_train)

        # Simpan scaler agar bisa dipakai live trading
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump(self.scaler, self.path)
        logger.info("✅ Scaler fitted on Training Data and saved.")

    def transform(self, df):
        """Transform data menggunakan scaler yang sudah di-fit"""
        if df.empty:
            return np.empty((0, len(self.feature_cols)))

        raw = df[self.feature_cols].values

        # Coba load scaler jika belum di-fit (untuk kasus live trading/inference)
        if not hasattr(self.scaler, "center_"):
            loaded = self.get_scaler()
            if loaded:
                self.scaler = loaded
            else:
                logger.warning(
                    "⚠️ Scaler not fitted! Scaling might be wrong for inference."
                )

        return self.scaler.transform(raw)

    def process(self, df, is_training=False):
        """Legacy wrapper agar kompatibel dengan kode lama (Inference Mode)"""
        df_processed = self.add_indicators(df)
        if df_processed.empty:
            return df_processed, np.empty((0, 4))

        if is_training:
            # Mode training lama (deprecated, sebaiknya pakai flow di train.py baru)
            self.fit_scaler(df_processed)
            scaled = self.transform(df_processed)
        else:
            scaled = self.transform(df_processed)

        return df_processed, scaled


class DataFetcher:
    async def fetch_market_data(self, days: int = 365, interval: str = "1h"):
        symbol = settings.YFINANCE_SYMBOL
        logger.info(f"📥 Downloading history for {symbol} ({days} days)...")
        try:
            ticker = yf.Ticker(symbol)

            df = ticker.history(period=f"{days}d", interval=interval)

            if df.empty:
                return pd.DataFrame()

            df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )
            df.index = df.index.tz_localize(None)
            df.index.name = "timestamp"
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return pd.DataFrame()


# Instance global
fetcher = DataFetcher()

processor = FeatureEngineer()
