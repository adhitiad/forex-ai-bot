import asyncio
import logging
import random
import time

import model
import numpy as np
import pandas as pd
import pandas_ta
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("DataFetcher")


SYMBOL = "BTC-USD"
PERIOD = "3y"
INTERVAL = "1h"


class DataFetcher:
    def __init__(self):
        self.last_call_time = 0
        self.MIN_DELAY = 60
        self.JITTER = 15

    async def smart_sleep(self):
        now = time.time()
        elapsed = now - self.last_call_time
        wait_time = max(0, self.MIN_DELAY - elapsed)
        random_jitter = random.uniform(0, self.JITTER)
        total_sleep = wait_time + random_jitter

        if total_sleep > 0:
            logger.info(
                "⏳ Rate Limit: Cooling down for %s and %s...",
                total_sleep,
                random_jitter,
            )
            await asyncio.sleep(total_sleep)
        self.last_call_time = time.time()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _download_safe(self, symbol, period, interval):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, proxy=None)
        if df.empty:
            raise ValueError("Empty Data")
        return df

    async def fetch_market_data(self, symbol=SYMBOL, period=PERIOD, interval=INTERVAL):
        """
        Fetches market data for the given symbol, period, and interval.

        Args:
            symbol (str): The trading symbol.
            period (str): The period for data.
            interval (str): The interval for data.

        Returns:
            pd.DataFrame: The market data dataframe.
        """
        await self.smart_sleep()
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(
                None, self._download_safe, symbol, period, interval
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
        except (ValueError, ConnectionError, TimeoutError) as e:
            logger.error("Fetch Error: %s", e)
            return pd.DataFrame()


fetcher = DataFetcher()


def process_features(df):
    if df.empty:
        return df, np.empty((0, 4))
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.dropna(inplace=True)
    cols = ["close", "RSI_14", "EMA_20", "ATRr_14"]
    if not all(c in df.columns for c in cols):
        return df, np.empty((0, 4))
    return df, df[cols].values


async def train_model_async(symbol: str, timeframe: str):
    """Asynchronous training process."""
    try:
        logger.info("Training started for %s with timeframe %s", symbol, timeframe)
        df, features = process_features(
            await fetcher.fetch_market_data(symbol, PERIOD, INTERVAL)
        )
        if df.empty:
            logger.warning("No valid data found for training.")
            return

        labels = features[:, 0]
        models = model.TimeSeriesTransformer(input_dim=features.shape[1])
        models = model.train_model(
            models, features, labels, epochs=18, batch_size=32, lr=0.001
        )

        await asyncio.sleep(1)
        logger.info("Training completed for %s", symbol)
    except (ValueError, RuntimeError, asyncio.TimeoutError) as e:
        logger.error("Training error: %s", e)
