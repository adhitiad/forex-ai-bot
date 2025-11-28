import asyncio
import logging
import random
import time

import numpy as np
import pandas as pd
import pandas_ta as ta
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
            logger.info(f"⏳ Rate Limit: Cooling down for {total_sleep:.2f}s...")
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
        except Exception as e:
            logger.error(f"Fetch Error: {e}")
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
