"""
AI Engine - Main Application
"""

import asyncio
import logging
import subprocess

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

from config import settings
from database import TradeLog, get_db, init_db
from stream_manager import streamor

app = FastAPI()
init_db()


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ConfigUpdate(BaseModel):
    asset_type: str
    symbol: str


@app.post("/api/control/config")
async def update_config(cfg: ConfigUpdate):
    settings.ASSET_TYPE = cfg.asset_type
    if cfg.asset_type == "CRYPTO":
        settings.SYMBOL_CRYPTO = cfg.symbol
    else:
        settings.SYMBOL_FOREX = cfg.symbol
    await streamor.push_control(cfg.dict())
    return {"status": "ok"}


@app.post("/api/control/train")
def train_model():
    subprocess.Popen(["python", "train.py"])
    return {"status": "started"}


@app.get("/api/control/trade-history")
def history(db=Depends(get_db)):
    return db.query(TradeLog).order_by(TradeLog.timestamp.desc()).limit(50).all()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not streamor or not streamor.r:
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
        await websocket.close()
        pass
