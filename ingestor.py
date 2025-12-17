# File: ingestor.py
import asyncio
import datetime
import logging

import grpc
import MetaTrader5 as mt5

from config import settings
from logging_config import setup_logger

# Jika masih mau simpan ke DB via gRPC, uncomment bagian gRPC
from protos import market_pb2, market_pb2_grpc
from stream_manager import streamor

logger = setup_logger("MT5-Ingestor")


class Ingestor:
    def __init__(self):
        self.active_symbols = settings.ACTIVE_SYMBOLS
        self.channel = grpc.aio.insecure_channel("data_service:50051")
        self.stub = market_pb2_grpc.MarketDataServiceStub(self.channel)

    def init_mt5(self):
        if not mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        ):
            logger.error("MT5 Init Failed")
            return False
        return True

    async def fetch_and_push(self):
        if not self.active_symbols:
            return

        for symbol in self.active_symbols:
            # Ambil Candle Terakhir (M1)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)

            if rates is None or len(rates) == 0:
                continue

            tick = rates[0]
            # Convert numpy types to python native types
            price = float(tick["close"])
            vol = float(tick["tick_volume"])
            ts = str(datetime.datetime.now())

            # Payload untuk Redis Stream (Brain)
            stream_payload = {
                "symbol": symbol,
                "close": price,
                "open": float(tick["open"]),
                "high": float(tick["high"]),
                "low": float(tick["low"]),
                "volume": vol,
                "timestamp": ts,
            }

            # Push ke Redis
            await streamor.push_market_data(symbol, stream_payload)

            # Log periodic (agar terminal tidak penuh)
            # logger.info(f"Tick: {symbol} @ {price}")

    async def run(self):
        await streamor.connect()
        if not self.init_mt5():
            return

        logger.info(f"📡 MT5 Ingestor Started. Watching: {self.active_symbols}")

        while True:
            await self.fetch_and_push()
            # Ambil data setiap 1 detik (Real-time)
            # Jauh lebih cepat dari yfinance (60s)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(Ingestor().run())
