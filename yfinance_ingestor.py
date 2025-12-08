import asyncio
import datetime
import logging
import yfinance as yf
from config import settings
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YFinanceIngestor")


class YFinanceIngestorEngine:
    """Ingestor menggunakan Yahoo Finance untuk live data"""

    def __init__(self):
        self.running = True
        self.symbol = settings.YFINANCE_SYMBOL
        self.ticker = None

    async def init_ticker(self):
        try:
            self.ticker = yf.Ticker(self.symbol)
            logger.info(f"✅ Connected to Yahoo Finance for {self.symbol}")
        except Exception as e:
            logger.error(f"❌ Init Error: {e}")

    async def run(self):
        await self.init_ticker()

        if self.ticker is None:
            logger.error("❌ Ticker not initialized. Exiting...")
            return

        logger.info(f"🚀 Tracking: {self.symbol}")

        retry_count = 0
        max_retries = 3

        while self.running:
            try:
                # Get current price info
                info = self.ticker.info

                # Get recent history for OHLCV
                hist = self.ticker.history(period="1d", interval="1m")

                if hist.empty:
                    logger.warning("⚠️ No data received")
                    await asyncio.sleep(5)
                    continue

                # Get latest candle
                latest = hist.iloc[-1]

                payload = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "close": float(latest["Close"]),
                    "volume": float(latest["Volume"]),
                    "source": "YAHOO_FINANCE",
                }

                await streamor.push_market_data(settings.ACTIVE_SYMBOL, payload)
                logger.info(f"🪙 {self.symbol}: ${payload['close']:.2f}")

                # Reset retry counter on success
                retry_count = 0
                await asyncio.sleep(10)  # Yahoo Finance has rate limits, poll every 10s

            except Exception as e:
                retry_count += 1
                wait_time = min(retry_count * 5, 30)  # Max 30 seconds
                logger.error(f"❌ Fetch Error (attempt {retry_count}): {e}")

                if retry_count >= max_retries:
                    logger.warning(f"⚠️ Max retries reached, resetting counter...")
                    retry_count = 0

                await asyncio.sleep(wait_time)


if __name__ == "__main__":
    asyncio.run(YFinanceIngestorEngine().run())
