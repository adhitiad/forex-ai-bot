import asyncio
import datetime
import logging
import ccxt.async_support as ccxt
from config import settings
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingestor")


class IngestorEngine:
    def __init__(self):
        self.running = True
        self.symbol = settings.ACTIVE_SYMBOL
        self.exchange = None

    async def init_exchange(self):
        try:
            exchange_class = getattr(ccxt, settings.EXCHANGE_ID)
            self.exchange = exchange_class(
                {
                    "apiKey": settings.TOKOCRYPTO_API_KEY,
                    "secret": settings.TOKOCRYPTO_SECRET,
                    "enableRateLimit": True,
                    "timeout": 30000,  # 30 seconds timeout
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True,
                    },
                }
            )
            # Don't call load_markets() - it can cause timeout issues
            # Markets will be loaded on-demand when needed
            logger.info(f"✅ Connected to {settings.EXCHANGE_ID.upper()}")
        except Exception as e:
            logger.error(f"❌ Init Error: {e}")
            # Don't stop running - try to continue with fetch_ticker
            # self.running = False

    async def run(self):
        await self.init_exchange()

        # Ensure exchange was successfully initialized
        if self.exchange is None:
            logger.error("❌ Exchange not initialized. Exiting...")
            return

        logger.info(f"🚀 Tracking: {self.symbol}")

        retry_count = 0
        max_retries = 3

        while self.running:
            try:
                ticker = await self.exchange.fetch_ticker(self.symbol)
                payload = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "open": float(
                        ticker["open"] if ticker.get("open") else ticker["last"]
                    ),
                    "high": float(
                        ticker["high"] if ticker.get("high") else ticker["last"]
                    ),
                    "low": float(
                        ticker["low"] if ticker.get("low") else ticker["last"]
                    ),
                    "close": float(ticker["last"]),
                    "volume": float(ticker["baseVolume"]),
                    "source": settings.EXCHANGE_ID.upper(),
                }
                await streamor.push_market_data(self.symbol, payload)
                logger.info(f"🪙 {self.symbol}: {payload['close']}")

                # Reset retry counter on success
                retry_count = 0
                await asyncio.sleep(2)

            except Exception as e:
                retry_count += 1
                wait_time = min(retry_count * 5, 30)  # Max 30 seconds
                logger.error(f"❌ Fetch Error (attempt {retry_count}): {e}")

                if retry_count >= max_retries:
                    logger.warning(f"⚠️ Max retries reached, resetting counter...")
                    retry_count = 0

                await asyncio.sleep(wait_time)

        if self.exchange:
            await self.exchange.close()


if __name__ == "__main__":
    asyncio.run(IngestorEngine().run())
