import asyncio
import datetime
import logging

from config import settings
from features import fetcher
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingestor-Forex")


class IngestorEngine:
    def __init__(self):
        self.running = True
        self.symbol = settings.ACTIVE_SYMBOL

    async def is_market_open(self):
        """Cek apakah pasar Forex buka (Senin-Jumat)"""
        now = datetime.datetime.now()
        # 0=Senin, 4=Jumat, 5=Sabtu, 6=Minggu
        if now.weekday() >= 5:
            return False
        return True

    async def data_producer(self):
        logger.info(f"🌍 Forex Engine Started. Target: {self.symbol}")

        # Setup fetcher ke mode Forex
        fetcher.update_config("FOREX", self.symbol)

        while self.running:
            try:
                # 1. Cek Libur Pasar
                if not await self.is_market_open():
                    logger.info("💤 Market Closed (Weekend). Sleeping for 10 hours...")
                    await asyncio.sleep(36000)  # Tidur sabtu dan minggu
                    continue

                # 2. Ambil Data
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
                        "source": "FOREX",
                    }

                    await streamor.push_market_data(self.symbol, payload)
                    logger.info(f"💱 Tick {self.symbol}: {payload['close']}")

                # Forex tidak secepat Crypto, polling tiap 60 detik cukup
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error fetching Forex data: {e}")
                await asyncio.sleep(10)

    async def run(self):
        # Kita hapus command_listener karena ini mode Khusus Forex (Hardcoded)
        await self.data_producer()


if __name__ == "__main__":
    asyncio.run(IngestorEngine().run())
