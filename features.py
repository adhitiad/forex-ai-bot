import asyncio
import logging
import os
import time
import joblib
import numpy as np
import pandas as pd
import pandas_ta
import ccxt.async_support as ccxt
from sklearn.preprocessing import RobustScaler
from config import settings

logger = logging.getLogger("Features")


class DataFetcher:
    def __init__(self):
        self.exchange = None

    async def get_exchange(self):
        if not self.exchange:
            exchange_class = getattr(ccxt, settings.EXCHANGE_ID)
            self.exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,  # 30 seconds timeout
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True,
                    },
                    # Kita tidak butuh API Key untuk download history candle (Data Publik)
                }
            )
        return self.exchange

    async def fetch_market_data(self, days: int = 365):
        """
        Download data historis Crypto menggunakan CCXT dengan Pagination.
        Mengambil data sebanyak 'days' hari ke belakang.
        """
        exchange = await self.get_exchange()
        symbol = settings.ACTIVE_SYMBOL
        timeframe = settings.TIMEFRAME

        logger.info(f"📥 Downloading history for {symbol} ({days} days)...")

        # Hitung waktu mulai (ms)
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        all_ohlcv = []

        max_retries = 3
        retry_count = 0

        try:
            while True:
                try:
                    # Download batch (biasanya limit 1000 candles per call)
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol, timeframe, since, limit=1000
                    )

                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)

                    # Update pointer waktu ke candle terakhir + 1
                    last_timestamp = ohlcv[-1][0]
                    since = last_timestamp + 1

                    logger.info(f"   ...Fetched {len(all_ohlcv)} candles so far")

                    # Jika sudah sampai waktu sekarang, stop
                    if last_timestamp >= exchange.milliseconds():
                        break

                    # Jeda sedikit biar gak kena ban
                    await asyncio.sleep(exchange.rateLimit / 1000)

                    # Reset retry counter on success
                    retry_count = 0

                except Exception as fetch_error:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise fetch_error

                    wait_time = retry_count * 2  # Exponential backoff
                    logger.warning(
                        f"⚠️ Fetch error (attempt {retry_count}/{max_retries}): {fetch_error}"
                    )
                    logger.info(f"   Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

            if not all_ohlcv:
                return pd.DataFrame()

            # Convert ke DataFrame
            df = pd.DataFrame(
                data=all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Hapus duplikat jika ada
            df = df.loc[~df.index.duplicated(keep="first")]

            logger.info(f"✅ Download Complete: {len(df)} rows.")
            await exchange.close()
            return df

        except Exception as e:
            logger.error(f"❌ Error downloading history: {e}")
            await exchange.close()
            return pd.DataFrame()


class FeatureEngineer:
    def __init__(self, path=settings.SCALER_FILE):
        self.path = path
        self.scaler = RobustScaler()
        # Fitur yang digunakan (RSI, EMA, ATR, Returns)
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
        """Generate Technical Indicators"""
        if df.empty:
            return df, np.empty((0, 4))

        df = df.copy()

        # Hitung Indikator
        df["returns"] = df["close"].pct_change()
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.atr(length=14, append=True)

        # Bersihkan NaN (akibat windowing indikator)
        df.dropna(inplace=True)

        if df.empty:
            return df, np.empty((0, 4))

        # Cek kelengkapan kolom
        if not all(c in df.columns for c in self.cols):
            return df, np.empty((0, 4))

        raw = df[self.cols].values

        # Scaling Logic
        if is_training:
            scaled = self.scaler.fit_transform(raw)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            joblib.dump(self.scaler, self.path)
            logger.info(f"💾 Scaler saved to {self.path}")
        else:
            if not hasattr(self.scaler, "center_"):
                if not self.load_scaler():
                    # Fallback jika scaler hilang saat live: fit sementara (tidak ideal tapi jalan)
                    scaled = self.scaler.fit_transform(raw)
                else:
                    scaled = self.scaler.transform(raw)
            else:
                scaled = self.scaler.transform(raw)

        return df, scaled


fetcher: DataFetcher = DataFetcher()
processor: FeatureEngineer = FeatureEngineer()
