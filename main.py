"""
AI Engine - Main Application (Secured)
"""

import asyncio
import logging
import json
import datetime
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Security, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from database import TradeLog, get_db, init_db
from stream_manager import streamor
from state_manager import state_manager

# Log Buffer
log_buffer = []


class ListHandler(logging.Handler):
    def emit(self, r):
        log_buffer.append(self.format(r))
        if len(log_buffer) > 100:
            log_buffer.pop(0)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainAPI")
logger.addHandler(ListHandler())

API_KEY_NAME = "X-Bot-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(key: str = Security(api_key_header)):
    if key == settings.API_SECRET or not settings.API_SECRET:
        return key
    raise HTTPException(403, "Invalid Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    streamor.connect()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ManualOrder(BaseModel):
    action: str
    symbol: str


@app.get("/api/dashboard/stats", dependencies=[Depends(get_api_key)])
async def get_stats(db: Session = Depends(get_db)):
    total = db.query(TradeLog).filter(TradeLog.status == "CLOSED").count()
    wins = (
        db.query(TradeLog)
        .filter(
            TradeLog.status == "CLOSED", (TradeLog.tp_price - TradeLog.entry_price) > 0
        )
        .count()
    )
    return {
        "total_trades": total,
        "win_rate": round((wins / total * 100) if total else 0, 2),
        "active_position": await state_manager.get_active_position(),
        "symbol": settings.ACTIVE_SYMBOL,
    }


@app.get("/api/system/logs")
def get_logs():
    return {"logs": log_buffer[::-1]}


@app.post("/api/control/manual-order", dependencies=[Depends(get_api_key)])
async def manual(order: ManualOrder):
    await streamor.push_signal(
        {
            "action": order.action.upper(),
            "symbol": order.symbol,
            "entry_price": 0,
            "status": "MANUAL",
            "timestamp": str(datetime.datetime.now()),
        }
    )
    return {"status": "sent"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not streamor.r:
        await websocket.close()
        return
    pubsub = streamor.r.pubsub()
    await pubsub.subscribe("channel_market", "channel_signals")
    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True)
            if msg:
                await websocket.send_json(
                    {"type": msg["channel"], "payload": json.loads(msg["data"])}
                )
            await asyncio.sleep(0.01)
    except:
        pass
