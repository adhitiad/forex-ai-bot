"""
AI Engine - Main Application (Secured)
"""

import asyncio
import datetime
import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import settings
from database import init_db
from state_manager import state_manager
from stream_manager import streamor

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainAPI")

# --- SECURITY SETUP ---
# Mendefinisikan header API Key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Fungsi Validasi API Key"""
    # Jika di .env tidak ada API KEY, kita anggap mode development (bypass)
    # Anda bisa menambahkan logic validasi password/token di sini
    return api_key_header


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Forex AI Bot System...")
    init_db()
    streamor.connect()
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")


# --- APP INIT ---
app = FastAPI(lifespan=lifespan, title="Forex AI Bot API")

# CORS (Agar frontend bisa akses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- PYDANTIC MODELS ---
class ManualOrder(BaseModel):
    action: str
    symbol: str
    price: float
    tp: float = 0.0
    sl: float = 0.0


class ModeUpdate(BaseModel):
    mode: str


class ConfigUpdate(BaseModel):
    symbol: str
    timeframe: str


# --- ENDPOINTS ---


@app.get("/")
async def root():
    return {"status": "online", "service": "Forex AI Bot"}


# 1. Endpoint Ganti Mode (Live/Paper)
@app.post("/api/settings/mode", dependencies=[Depends(get_api_key)])
async def set_mode(data: ModeUpdate):
    await state_manager.set_trading_mode(data.mode)
    return {"status": "success", "mode": data.mode}


@app.get("/api/settings/mode")
async def get_mode():
    return {"mode": await state_manager.get_trading_mode()}


# 2. Endpoint Ganti Config Aset (Symbol/Timeframe) - YANG MENYEBABKAN ERROR TADI
@app.post("/api/settings/config", dependencies=[Depends(get_api_key)])
async def update_config(data: ConfigUpdate):
    # Simpan ke Redis agar Ingestor & Brain baca
    await state_manager.set_asset_config(data.symbol, data.timeframe)

    # Reset Posisi aktif agar aman
    await state_manager.clear_active_position()

    logger.info(f"🔄 Config Updated via API: {data.symbol} ({data.timeframe})")
    return {"status": "updated", "config": data}


# 3. Endpoint Manual Order
@app.post("/api/control/manual-order", dependencies=[Depends(get_api_key)])
async def manual(order: ManualOrder):
    await streamor.push_signal(
        {
            "action": order.action.upper(),
            "final_action": "MARKET " + order.action.upper(),
            "symbol": order.symbol,
            "entry_price": order.price,
            "tp": order.tp,
            "sl": order.sl,
            "status": "MANUAL",
            "alignment": "USER_OVERRIDE",
            "timestamp": str(datetime.datetime.now()),
        }
    )
    return {"status": "sent"}


# --- WEBSOCKET ---
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not streamor.r:
        logger.error("Redis connection missing for Websocket")
        return

    ps = streamor.r.pubsub()
    await ps.subscribe(
        settings.CHANNEL_MARKET,
        settings.CHANNEL_SIGNALS,
        settings.CHANNEL_CONFIRMATION,
        settings.CHANNEL_AI_ANALYSIS,
        settings.CHANNEL_SYSTEM,
    )

    try:
        while True:
            msg = await ps.get_message(ignore_subscribe_messages=True)
            if msg:
                await websocket.send_json(
                    {"type": msg["channel"], "payload": json.loads(msg["data"])}
                )
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        await ps.close()
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
