import asyncio
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

from config import settings

logger = logging.getLogger("Features")


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(series, period=20):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


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


class FeatureEngineer:
    def __init__(self, path=settings.SCALER_FILE):
        self.path = path
        self.scaler = RobustScaler()
        self.cols = ["returns", "RSI_14", "EMA_20", "ATRr_14"]

    def load_scaler(self):
        if os.path.exists(self.path):
            try:
                self.scaler = joblib.load(self.path)
                return True
            except:
                return False
        return False

    def process(self, df, is_training=False):
        if df.empty:
            return df, np.empty((0, 4))
        df = df.copy()

        df["returns"] = df["close"].pct_change()
        df["RSI_14"] = calculate_rsi(df["close"], period=14)
        df["EMA_20"] = calculate_ema(df["close"], period=20)
        df["ATRr_14"] = calculate_atr(df, period=14)
        df.dropna(inplace=True)

        if df.empty:
            return df, np.empty((0, 4))
        raw = df[self.cols].values

        if is_training:
            scaled = self.scaler.fit_transform(raw)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            joblib.dump(self.scaler, self.path)
        else:
            if not hasattr(self.scaler, "center_"):
                if not self.load_scaler():
                    scaled = self.scaler.fit_transform(raw)
                else:
                    scaled = self.scaler.transform(raw)
            else:
                scaled = self.scaler.transform(raw)

        return df, scaled


fetcher = DataFetcher()
processor = FeatureEngineer()
