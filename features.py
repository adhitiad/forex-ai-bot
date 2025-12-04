import asyncio
import logging
import os
import time

import ccxt
import joblib
import numpy as np
import pandas as pd
import pandas_ta
import yfinance as yf
from sklearn.preprocessing import RobustScaler

from config import settings

logger = logging.getLogger("Features")


class DataFetcher:
    def __init__(self):
        self.last_call = 0
        self.exchange = None
        self.current_asset_type = settings.ASSET_TYPE
        self.current_symbol = settings.ACTIVE_SYMBOL
        self._init_exchange()

    def _init_exchange(self):
        if self.current_asset_type == "CRYPTO":
            try:
                exchange_class = getattr(ccxt, settings.EXCHANGE_ID)
                self.exchange = exchange_class({"enableRateLimit": True})
            except:
                pass
        else:
            if self.exchange:
                try:
                    asyncio.create_task(
                        self.exchange.close()
                    )  # pylint: disable=no-member
                except AttributeError:
                    pass
            self.exchange = None

    def update_config(self, asset_type, symbol):
        self.current_asset_type = asset_type
        self.current_symbol = symbol
        self._init_exchange()

    async def fetch_market_data(self, period="5y", limit=3000):
        if self.current_asset_type == "CRYPTO":
            if not self.exchange:
                return pd.DataFrame()
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    self.current_symbol, timeframe=settings.TIMEFRAME, limit=limit
                )
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            except:
                return pd.DataFrame()
        else:
            elapsed = time.time() - self.last_call
            if elapsed < 2:
                await asyncio.sleep(2 - elapsed)
            self.last_call = time.time()
            try:
                df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: yf.download(
                        self.current_symbol,
                        period=period,
                        interval=settings.TIMEFRAME,
                        progress=False,
                    ),
                )
                if df is None or df.empty:
                    return pd.DataFrame()
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
            except:
                return pd.DataFrame()

    async def close(self):
        if self.exchange:
            await self.exchange.close()


class FeatureEngineer:
    def __init__(self, path=settings.SCALER_FILE):
        self.path = path
        self.scaler = RobustScaler()
        self.cols = ["returns", "RSI_14", "EMA_20", "ATRr_14"]

    def load_scaler(self):
        if os.path.exists(self.path):
            self.scaler = joblib.load(self.path)
            return True
        return False

    def process(self, df, is_training=False):
        if df.empty:
            return df, np.empty((0, 4))
        df = df.copy()
        df["returns"] = df["close"].pct_change()
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
