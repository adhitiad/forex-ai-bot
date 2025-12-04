import asyncio
import logging

from config import settings
from features import fetcher
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingestor 🚘🚘🚘")


class IngestorEngine:
    def __init__(self):
        self.running = True
        self.symbol = settings.ACTIVE_SYMBOL
        self.asset_type = settings.ASSET_TYPE

    async def command_listener(self):
        async for cmd in streamor.listen_control():
            if "asset_type" in cmd and "symbol" in cmd:
                self.asset_type = cmd["asset_type"]
                self.symbol = cmd["symbol"]
                fetcher.update_config(self.asset_type, self.symbol)
                logger.info(f"🔄 Switched to {self.symbol}")

    async def data_producer(self):
        logger.info(f"🔌 Started. Target: {self.symbol}")
        while self.running:
            try:
                df = await fetcher.fetch_market_data(period="1d", limit=100)
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    payload = {
                        "timestamp": str(df.index[-1]),
                        "open": float(last["open"]),
                        "high": float(last["high"]),
                        "low": float(last["low"]),
                        "close": float(last["close"]),
                        "volume": float(last["volume"]),
                        "source": self.asset_type,
                    }
                    await streamor.push_market_data(self.symbol, payload)
                    logger.info(f"Tick: {payload['close']}")
                await asyncio.sleep(10 if self.asset_type == "CRYPTO" else 60)
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(5)

    async def run(self):
        await asyncio.gather(self.command_listener(), self.data_producer())


if __name__ == "__main__":
    asyncio.run(IngestorEngine().run())
