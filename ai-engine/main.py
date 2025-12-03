"""
AI Engine - Main Application
"""

import asyncio
import json
import os
import subprocess
from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

try:
    from database import ModelLog, TradeLog, get_db

    DATABASE_AVAILABLE = True
except Exception as e:
    print(f"Database not available: {e}")
    DATABASE_AVAILABLE = False

try:
    from stream_manager import streamor

    REDIS_AVAILABLE = True
except Exception as e:
    print(f"Redis not available: {e}")
    REDIS_AVAILABLE = False

app = FastAPI()

# --- ROUTES ---
control = APIRouter(prefix="/api/control")
chart = APIRouter(prefix="/api/chart")


@control.post("/train")
async def train_model(bg: BackgroundTasks):
    if DATABASE_AVAILABLE:
        try:
            with get_db() as db:
                db.add(ModelLog(version="auto", status="STARTED"))
                db.commit()
        except Exception as e:
            print(f"Database error: {e}")

    bg.add_task(lambda: subprocess.run(["python", "train.py"]))
    return {"status": "started"}


@control.get("/trade-history")
async def history():
    if DATABASE_AVAILABLE:
        try:
            with get_db() as db:
                return (
                    db.query(TradeLog)
                    .order_by(TradeLog.timestamp.desc())
                    .limit(50)
                    .all()
                )
        except Exception as e:
            print(f"Database error: {e}")
            return []
    return []


@chart.get("/history")
async def chart_history():
    if REDIS_AVAILABLE:
        try:
            raw = streamor.r.xrevrange(streamor.STREAM_MARKET, "+", "-", count=100)
            return [json.loads(v["data"]) for _, v in reversed(raw) if "data" in v]
        except Exception as e:
            print(f"Redis error: {e}")
    return []


app.include_router(control)
app.include_router(chart)

# --- WEBSOCKET ---
clients = []


@app.on_event("startup")
async def start_listener():
    if REDIS_AVAILABLE:
        asyncio.create_task(redis_listener())


async def redis_listener():
    if not REDIS_AVAILABLE:
        return

    try:
        pubsub = streamor.r.pubsub()
        pubsub.subscribe("channel_signals")
        while True:
            message = await asyncio.to_thread(pubsub.get_message, timeout=1.0)
            if message and message["type"] == "message":
                for client in clients:
                    await client.send_text(message["data"])
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Redis pubsub error: {e}")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except:
        clients.remove(ws)
