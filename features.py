import asyncio
import logging
import os
import time
from math import e

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
        # Fitur yang digunakan (Pastikan urutannya KONSISTEN selamanya)
        self.cols = ["returns", "RSI_14", "EMA_20", "ATRr_14"]

    def load_scaler(self):
        """Memuat scaler dari disk agar normalisasi data live sama dengan data training"""
        if os.path.exists(self.path):
            try:
                self.scaler = joblib.load(self.path)
                return True
            except Exception as e:
                logging.error(f"Error loading scaler: {e}")
                return False
        return False

    def process(self, df, is_training=False):
        """
        Mengubah OHLCV menjadi Technical Indicators
        is_training=True -> Menghitung statistik scaling baru dan SIMPAN ke file.
        is_training=False -> Menggunakan statistik scaling yang sudah disimpan.
        """
        if df.empty:
            return df, np.empty((0, 4))

        df = df.copy()

        # 1. Technical Indicators Calculation
        df["returns"] = df["close"].pct_change()
        # Gunakan fillna untuk menghindari NaN di awal jika memungkinkan,
        # tapi dropna lebih aman untuk akurasi model
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.atr(length=14, append=True)

        # Hapus baris NaN akibat perhitungan indikator (20 baris pertama)
        df.dropna(inplace=True)

        if df.empty:
            return df, np.empty((0, 4))

        # Pastikan kolom tersedia
        if not all(c in df.columns for c in self.cols):
            logging.error(f"Missing columns. Found: {df.columns}")
            return df, np.empty((0, 4))

        raw = df[self.cols].values

        # 2. Scaling (RobustScaler untuk menangani outlier crypto/forex)
        if is_training:
            # Mode Training: Pelajari data dan simpan "kamus" scaling
            scaled = self.scaler.fit_transform(raw)
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            joblib.dump(self.scaler, self.path)
            logging.info(f"Scaler saved to {self.path}")
        else:
            # Mode Live/Brain: Gunakan "kamus" lama
            # Jangan fit ulang! Nanti bot bingung nilai 0.5 itu berapa.
            try:
                # Cek atribut scaler untuk memastikan sudah di-fit
                if not hasattr(self.scaler, "center_"):
                    if not self.load_scaler():
                        logging.error("Scaler not fitted and file not found!")
                        return df, np.empty((0, 4))
                scaled = self.scaler.transform(raw)
            except Exception as e:
                logging.error(f"Scaling error: {e}")
                return df, np.empty((0, 4))

        return df, scaled

    def save_to_dataset(self, df, folder):
        """Simpan DataFrame ke dalam format dataset (CSV) di folder tertentu."""
        if df.empty:
            logging.warning("DataFrame is empty. Skipping save.")
            return
        elif not df.columns:
            logging.warning("DataFrame has no columns. Skipping save.")
            return

        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{settings.ACTIVE_SYMBOL}_dataset.csv")
        df.to_csv(file_path, index=False)
        logging.info(f"Dataset saved to {file_path}")


# Instance global
fetcher = DataFetcher()  # Pastikan class DataFetcher didefinisikan/diimport
processor = FeatureEngineer()
