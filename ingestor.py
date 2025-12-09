import asyncio
import datetime
import logging

import yfinance as yf

from config import settings
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YFinanceIngestor")


class YFinanceIngestor:
    def __init__(self):
        self.symbol = settings.ACTIVE_SYMBOL
        self.interval = settings.TIMEFRAME

    async def fetch_candle(self):
        try:
            # Ambil data intraday terbaru (period=1y cukup untuk intraday)
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period="1y", interval=self.interval)

            if not df.empty:
                # Ambil candle terakhir
                latest = df.iloc[-1]

                # Format payload sesuai standar bot
                payload = {
                    "timestamp": str(latest.name),
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "close": float(latest["Close"]),
                    "volume": float(latest["Volume"]),
                    "source": "YFINANCE",
                }

                await streamor.push_market_data(self.symbol, payload)
                logger.info(f"📈 {self.symbol} | Close: {payload['close']:.5f}")
            else:
                logger.warning("⚠️ Data kosong dari Yahoo Finance (Market Tutup?)")

        except Exception as e:
            logger.error(f"❌ YFinance Error: {e}")

    async def run(self):
        logger.info(f"🚀 YFinance Ingestor Started: {self.symbol} [{self.interval}]")
        streamor.connect()
        while True:
            await self.fetch_candle()
            # Polling setiap 60 detik (Yahoo Finance delay 1-2 menit, jangan spam)
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(YFinanceIngestor().run())
