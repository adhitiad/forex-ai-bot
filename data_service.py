import asyncio
import datetime
import logging
from concurrent import futures

import grpc

from config import settings
from database import MarketTick, get_db, init_db

# Import hasil generate proto
from protos import market_pb2, market_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataService-GRPC")


class MarketDataServicer(market_pb2_grpc.MarketDataServiceServicer):
    def SubmitTick(self, request, context):
        """Menerima 1 Tick dan simpan ke DB"""
        db = next(get_db())
        try:
            # Konversi timestamp string ke object datetime
            # Asumsi format ISO dari yfinance
            try:
                ts = datetime.datetime.fromisoformat(request.timestamp)
            except:
                ts = datetime.datetime.utcnow()

            new_tick = MarketTick(
                time=ts,
                symbol=request.symbol,
                price=request.price,
                volume=request.volume,
            )
            db.add(new_tick)
            db.commit()
            return market_pb2.TickResponse(success=True, message="Saved to TimescaleDB")
        except Exception as e:
            logger.error(f"DB Error: {e}")
            return market_pb2.TickResponse(success=False, message=str(e))
        finally:
            db.close()


async def serve():
    init_db()  # Pastikan Hypertable siap

    server = grpc.aio.server()
    market_pb2_grpc.add_MarketDataServiceServicer_to_server(
        MarketDataServicer(), server
    )

    # Listen di Port gRPC
    server.add_insecure_port(settings.GRPC_SERVER_HOST)
    logger.info(f"🚀 gRPC Data Service running on {settings.GRPC_SERVER_HOST}")
    logger.info("🔥 Connected to TimescaleDB Cloud")

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
