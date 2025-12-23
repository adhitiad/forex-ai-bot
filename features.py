import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import settings

logger = logging.getLogger("Features")


class FeatureEngineer:
    def __init__(self):
        # UPDATE: Input Dimension sekarang = 7
        self.feature_cols = [
            "returns",
            "RSI_14",
            "EMA_20",
            "ATRr_14",
            "hour_sin",
            "hour_cos",
            "gap_jump",
        ]

    def add_indicators(self, df):
        """Menghitung indikator teknikal + Time Features"""
        if df.empty:
            return df
        df = df.copy()

        # 1. Basic Indicators
        df["returns"] = df["close"].pct_change()
        # RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # EMA 20
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

        # ATR 14
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        df["ATRr_14"] = tr.rolling(window=14).mean()

        # --- 2. FITUR KHUSUS SAHAM & FOREX (Time Encoding) ---
        # Ini membantu AI membedakan 'Awal Pasar', 'Tengah Pasar', 'Akhir Pasar'
        # Tanpa peduli tanggalnya kapan.

        # Ambil jam dari kolom time
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        hours = df["time"].dt.hour

        # Ubah jadi siklus lingkaran (Sinus & Cosinus)
        # Jam 23 akan dekat dengan jam 0, Jam 16 (Tutup saham) punya pola sendiri
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        # 3. GAP DETECTION (Lonjakan Harga)
        # Menghitung % loncatan dari Close candle sebelumnya ke Open candle sekarang
        df["prev_close"] = df["close"].shift(1)
        df["gap_jump"] = (df["open"] - df["prev_close"]) / df["prev_close"]
        # --- 3. FITUR GAP DETECTION (Penting untuk Saham) ---
        # Mengukur loncatan harga dari Close candle sebelumnya ke Open candle sekarang
        # Di saham, ini mendeteksi Gap Up/Down pagi hari

        # Hapus baris NaN (biasanya baris pertama)
        df.dropna(inplace=True)
        return df

    def fit_scaler(self, df, symbol):
        """Fit scaler khusus untuk symbol tertentu"""
        if df.empty:
            return

        scaler = RobustScaler()
        # Pastikan kolom yang diambil sesuai list baru (7 kolom)
        raw_data = df[self.feature_cols].values
        scaler.fit(raw_data)

        path = settings.get_scaler_path(symbol)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(scaler, path)
        logger.info(f"✅ Scaler (Dim 7) saved for {symbol}")

    def transform(self, df, symbol):
        if df.empty:
            return np.empty((0, len(self.feature_cols)))

        path = settings.get_scaler_path(symbol)
        if not os.path.exists(path):
            logger.error(f"❌ Scaler for {symbol} not found! Run training first.")
            return np.empty((0, len(self.feature_cols)))

        scaler = joblib.load(path)
        return scaler.transform(df[self.feature_cols].values)

    def process(self, df, symbol, is_training=False):
        df_processed = self.add_indicators(df)
        if df_processed.empty:
            return df_processed, np.empty((0, 7))  # Return empty dim 7

        if is_training:
            self.fit_scaler(df_processed, symbol)

        scaled_data = self.transform(df_processed, symbol)
        return df_processed, scaled_data


processor = FeatureEngineer()
