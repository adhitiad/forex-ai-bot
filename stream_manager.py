"""
Telegram Stream Manager
"""

import asyncio
import json
import logging
from datetime import datetime

import pandas as pd
import redis.asyncio as redis
import yfinance as yf

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Stream-YFinance")


class MarketStreamer:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )

    async def connect(self):
        logger.info("✅ YFinance Streamer Active (Batch Polling Mode)")

    async def consume_market_data(self):
        """
        Mengambil data live.
        Teknik: Batch Download (Semua simbol sekali request) biar cepat.
        """
        candles = []
        tickers_str = " ".join(settings.ACTIVE_SYMBOLS)

        try:
            # Download data hari ini (1d), interval sesuai config (15m)
            # group_by='ticker' agar format rapi per simbol
            data = await asyncio.to_thread(
                yf.download,
                tickers=tickers_str,
                period="1d",
                interval=settings.TIMEFRAME,
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                threads=True,
            )

            if data.empty:
                return []

            # Loop per simbol untuk ambil baris terakhir
            for symbol in settings.ACTIVE_SYMBOLS:
                try:
                    # Handle jika cuma 1 simbol, format dataframe beda
                    if len(settings.ACTIVE_SYMBOLS) == 1:
                        df_sym = data
                    else:
                        df_sym = data[symbol]

                    # Bersihkan NaN
                    df_sym = df_sym.dropna()

                    if not df_sym.empty:
                        last_row = df_sym.iloc[-1]

                        # Masukkan ke format standar Bot
                        candles.append(
                            {
                                "symbol": symbol,
                                "open": float(last_row["Open"]),
                                "high": float(last_row["High"]),
                                "low": float(last_row["Low"]),
                                "close": float(last_row["Close"]),
                                "volume": float(last_row["Volume"]),
                                "timestamp": str(last_row.name),  # Jam Candle
                                "source": "YFINANCE",
                            }
                        )
                except KeyError:
                    continue  # Simbol mungkin gagal download, skip aja

            # Jeda Polling
            # Jika Timeframe 15m, tidak perlu cek tiap detik.
            # Cek tiap 1 menit sudah cukup cepat.
            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Stream Error: {e}")
            await asyncio.sleep(10)  # Jeda kalau error

        return candles

    async def push_signal(self, payload):
        await self.r.publish(settings.CHANNEL_AI_ANALYSIS, json.dumps(payload))


streamor = MarketStreamer()
