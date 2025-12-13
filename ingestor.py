import asyncio
import json
import logging

import grpc
import yfinance as yf

from config import settings
from database import UserPreference, get_db

# Import gRPC stubs
from protos import market_pb2, market_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingestor-GRPC")


class GrpcIngestor:
    def __init__(self):
        self.active_symbols = set()
        # Buka koneksi gRPC ke Data Service
        self.channel = grpc.aio.insecure_channel(settings.GRPC_SERVER_HOST)
        self.stub = market_pb2_grpc.MarketDataServiceStub(self.channel)

    def refresh_symbol_list(self):
        db = next(get_db())
        results = (
            db.query(UserPreference.active_symbol)
            .filter(UserPreference.is_active == True)
            .distinct()
            .all()
        )
        self.active_symbols = {r[0] for r in results}
        if not self.active_symbols:
            self.active_symbols = {settings.YFINANCE_SYMBOL}

    async def fetch_and_push(self):
        if not self.active_symbols:
            return
        tickers = " ".join(self.active_symbols)

        try:
            data = yf.download(
                tickers, period="1d", interval="1m", group_by="ticker", progress=False
            )

            for symbol in self.active_symbols:
                df = data if len(self.active_symbols) == 1 else data.get(symbol)
                if df is None or df.empty:
                    continue

                latest = df.iloc[-1]

                # --- KIRIM VIA GRPC (Microservice Call) ---
                # Ini jauh lebih efisien & terstruktur daripada JSON string di Redis
                request = market_pb2.TickRequest(
                    symbol=symbol,
                    price=float(latest["Close"]),
                    volume=float(latest["Volume"]),
                    timestamp=str(latest.name),
                )

                # Call Remote Procedure
                response = await self.stub.SubmitTick(request)

                if not response.success:
                    logger.error(f"gRPC Failed for {symbol}: {response.message}")

            logger.info(
                f"✅ Sent {len(self.active_symbols)} ticks to TimescaleDB via gRPC"
            )

        except Exception as e:
            logger.error(f"Ingest Error: {e}")

    async def run(self):
        while True:
            self.refresh_symbol_list()
            await self.fetch_and_push()
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(GrpcIngestor().run())
