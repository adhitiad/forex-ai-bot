import asyncio
import logging
import time

from config import settings
from features import fetcher
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)


async def run():
    logging.info("🔌 Ingestor Started: %s", settings.SYMBOL_YF)
    while True:
        try:
            df = await fetcher.fetch_market_data(settings.SYMBOL_YF)
            if not df.empty:
                last = df.iloc[-1]
                payload = {
                    "timestamp": str(df.index[-1]),
                    "open": float(last["open"]),
                    "high": float(last["high"]),
                    "low": float(last["low"]),
                    "close": float(last["close"]),
                    "volume": float(last["volume"]),
                }
                streamor.push_market_data(settings.SYMBOL_YF, payload)
                logging.info(f"Tick: {payload['close']}")
            await asyncio.sleep(60)
        except Exception as e:
            logging.error(f"Error: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(run())
