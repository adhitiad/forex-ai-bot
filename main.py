"""
AI Engine - Main Application (Secured)
"""

import asyncio
import logging
import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Security,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from config import settings
from database import TradeLog, get_db, init_db
from stream_manager import streamor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

# --- SECURITY SETUP ---
API_KEY_NAME = "X-Bot-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    # Validasi header request dengan API_SECRET di .env
    if api_key_header == settings.API_SECRET:
        return api_key_header
    # Jika API_SECRET kosong di .env, izinkan akses (Mode Debug/Bahaya)
    if not settings.API_SECRET:
        logger.warning("⚠️ API_SECRET not set! Endpoint is unsecured.")
        return "unsecured"
    raise HTTPException(status_code=403, detail="Invalid Authentication")


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    logger.info("✅ Database Initialized")
    yield
    # Shutdown
    logger.info("🛑 Shutting down API")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ConfigUpdate(BaseModel):
    asset_type: str
    symbol: str


# --- SECURED ENDPOINTS (Perlu API Key) ---
@app.post("/api/control/config", dependencies=[Depends(get_api_key)])
async def update_config(cfg: ConfigUpdate):
    settings.ASSET_TYPE = cfg.asset_type
    if cfg.asset_type == "CRYPTO":
        settings.SYMBOL_CRYPTO = cfg.symbol
    else:
        settings.SYMBOL_FOREX = cfg.symbol

    # Broadcast perubahan config ke Brain/Ingestor via Redis
    await streamor.push_control(cfg.dict())
    return {"status": "ok", "message": f"Switched to {cfg.symbol}"}


@app.post("/api/control/train", dependencies=[Depends(get_api_key)])
def train_model():
    # Menjalankan training di background process terpisah
    subprocess.Popen(["python", "train.py"])
    return {"status": "started", "message": "Training process launched in background"}


# --- PUBLIC ENDPOINTS (Read Only) ---
@app.get("/api/control/trade-history")
def history(db=Depends(get_db)):
    return db.query(TradeLog).order_by(TradeLog.timestamp.desc()).limit(50).all()


@app.get("/")
def health_check():
    return {"status": "running", "symbol": settings.ACTIVE_SYMBOL}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not streamor.r:
        await websocket.close()
        return

    pubsub = streamor.r.pubsub()
    await pubsub.subscribe("channel_market", "channel_signals", "channel_sentiment")

    try:
        async for m in pubsub.listen():
            if m["type"] == "message":
                await websocket.send_text(m["data"])
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
