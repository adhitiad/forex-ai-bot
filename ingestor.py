import asyncio
import logging

import grpc
import yfinance as yf

from config import settings
from logging_config import setup_logger
from protos import market_pb2, market_pb2_grpc
from stream_manager import streamor

logger = setup_logger("Ingestor")


class Ingestor:
    def __init__(self):
        self.active_symbols = settings.ACTIVE_SYMBOLS
        # Koneksi gRPC ke Data Service
        # Hostname 'data_service' didapat dari docker-compose
        self.channel = grpc.aio.insecure_channel("data_service:50051")
        self.stub = market_pb2_grpc.MarketDataServiceStub(self.channel)

    async def fetch_and_push(self):
        if not self.active_symbols:
            return

        tickers = " ".join(self.active_symbols)
        try:
            # Download Multi-Symbol sekaligus
            data = yf.download(
                tickers,
                period="1d",
                interval="1m",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            for symbol in self.active_symbols:
                # Handle single vs multi symbol dataframe structure
                df = data if len(self.active_symbols) == 1 else data.get(symbol)

                if df is None or df.empty:
                    continue

                # Ambil candle terakhir (Live Tick Simulation)
                latest = df.iloc[-1]
                price = float(latest["Close"])
                vol = float(latest["Volume"])
                ts = str(latest.name)

                # 1. Kirim ke Database (Archive) via gRPC
                try:
                    req = market_pb2.TickRequest(
                        symbol=symbol, price=price, volume=vol, timestamp=ts
                    )
                    # Fire & Forget (biar cepat, gak nunggu response)
                    asyncio.create_task(self.stub.SubmitTick(req))
                except Exception as e:
                    logger.error(f"gRPC Push Error {symbol}: {e}")

                # 2. Kirim ke Redis Stream (Untuk Brain AI Real-time)
                stream_payload = {
                    "symbol": symbol,
                    "close": price,
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "volume": vol,
                    "timestamp": ts,
                }
                await streamor.push_market_data(symbol, stream_payload)

            logger.info(f"✅ Ingested ticks for {len(self.active_symbols)} symbols")

        except Exception as e:
            logger.error(f"Ingest Loop Error: {e}")

    async def run(self):
        await streamor.connect()  # Init Redis connection
        logger.info(f"📡 Ingestor Started. Watching: {self.active_symbols}")
        while True:
            await self.fetch_and_push()
            await asyncio.sleep(60)  # Fetch tiap 1 menit (limit yfinance)


if __name__ == "__main__":
    asyncio.run(Ingestor().run())
