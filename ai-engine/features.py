import asyncio
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import pandas_ta
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("Features")


class DataFetcher:
    def __init__(self):
        self.last_call = 0

    async def fetch_market_data(self, symbol="EURUSD=X", period="5d", interval="1h"):
        elapsed = time.time() - self.last_call
        if elapsed < 10:
            await asyncio.sleep(10 - elapsed)
        self.last_call = time.time()

        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(
                None, self._download, symbol, period, interval
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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
            return df
        except Exception as e:
            logger.error(f"Fetch Error: {e}")
            return pd.DataFrame()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _download(self, s, p, i):
        df = yf.download(s, period=p, interval=i, progress=False)
        if df is None or df.empty:
            raise ValueError("Empty Data")
        return df


class FeatureEngineer:
    def __init__(self, path="scaler.pkl"):
        self.path = path
        self.scaler = RobustScaler()
        self.cols = ["close", "RSI_14", "EMA_20", "ATRr_14"]

    def load_scaler(self):
        if os.path.exists(self.path):
            self.scaler = joblib.load(self.path)
            return True
        return False

    def process(self, df, is_training=False):
        if df.empty:
            return df, np.empty((0, 4))
        df = df.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.atr(length=14, append=True)
        df.dropna(inplace=True)

        if not all(c in df.columns for c in self.cols):
            return df, np.empty((0, 4))

        raw = df[self.cols].values
        if is_training:
            scaled = self.scaler.fit_transform(raw)
            joblib.dump(self.scaler, self.path)
        else:
            try:
                scaled = self.scaler.transform(raw)
            except:
                return df, np.empty((0, 4))
        return df, scaled


fetcher = DataFetcher()
processor = FeatureEngineer()
