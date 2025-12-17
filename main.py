"""
AI Engine - Enterprise API Gateway
Kompatibel dengan: FBS MT5 Executor & Enterprise Fusion Engine
"""

import asyncio
import datetime
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Security, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import settings
from state_manager import state_manager
from stream_manager import streamor

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API-Gateway")

# --- SECURITY SETUP ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validasi API Key sederhana"""
    # Di production, ganti ini dengan validasi database/env yang aman
    # if api_key_header != settings.API_SECRET:
    #     raise HTTPException(status_code=403, detail="Invalid Key")
    return api_key_header


# --- REDIS CONNECTION ---
r_api = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global r_api
    logger.info("🚀 Starting Enterprise API Gateway...")
    r_api = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )
    await streamor.connect()
    yield
    # Shutdown
    logger.info("🛑 Shutting down API...")
    if r_api:
        await r_api.close()


app = FastAPI(lifespan=lifespan, title="Forex AI Enterprise API")

# CORS (Agar Dashboard Streamlit/Web bisa akses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODELS ---
class ManualOrder(BaseModel):
    action: str  # BUY, SELL
    symbol: str
    risk: float = 0.01  # Risk manual (default 1%)
    reason: str = "Manual API User"


class EmergencyAction(BaseModel):
    confirm: bool


class ConfigUpdate(BaseModel):
    mode: Optional[str] = None  # LIVE / PAPER
    max_risk: Optional[float] = None


# --- ENDPOINTS ---


@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Forex AI Enterprise",
        "symbol": settings.ACTIVE_SYMBOLS[0],
        "executor": "MT5",
    }


# 1. STATUS & DASHBOARD DATA
@app.get("/api/monitor/status", dependencies=[Depends(get_api_key)])
async def get_system_status():
    """Mengambil snapshot kesehatan sistem dari Redis"""
    if not r_api:
        return {"error": "Redis unavailable"}

    # Ambil data voting terakhir
    llm_raw = await r_api.get("llm:analysis")
    macro_status = await r_api.get("macro:status")

    # Ambil saldo terakhir dari MT5 Executor
    balance = await r_api.get("account_balance")
    equity = await r_api.get("account_equity")

    return {
        "macro": macro_status,
        "llm_analysis": json.loads(llm_raw) if llm_raw else None,
        "account": {"balance": float(balance or 0), "equity": float(equity or 0)},
    }


# 2. MANUAL ORDER (Masuk ke Fusion Engine)
@app.post("/api/trade/manual", dependencies=[Depends(get_api_key)])
async def manual_trade(order: ManualOrder):
    """
    Kirim sinyal manual.
    NOTE: Sinyal ini akan masuk ke CHANNEL_SIGNALS dan dinilai oleh Fusion Engine.
    Jika Macro status 'DANGER', order ini mungkin tetap ditolak demi keamanan (kecuali dipaksa).
    """
    payload = {
        "source": "USER_MANUAL",  # Fusion Engine akan memberi bobot tinggi utk source ini
        "action": order.action.upper(),
        "symbol": order.symbol,
        "confidence": 1.0,  # Keyakinan penuh user
        "reason": order.reason,
        "timestamp": str(datetime.datetime.now()),
    }

    # Push ke Redis Channel Signals (Input Fusion)
    await streamor.push_signal(payload)

    logger.info(f"👨‍💻 Manual Order Sent: {order.action} {order.symbol}")
    return {"status": "queued", "message": "Signal sent to Fusion Engine"}


# 3. EMERGENCY (Bypass Fusion, Langsung ke Executor)
@app.post("/api/control/emergency-close", dependencies=[Depends(get_api_key)])
async def emergency_close(action: EmergencyAction):
    """PANIC BUTTON: Memaksa Executor menutup semua posisi."""
    if not action.confirm:
        return {"status": "aborted", "message": "Confirm param required"}

    # Payload khusus yang dimengerti Executor (Perlu update logic executor sedikit untuk handle ini)
    # Atau kita kirim SELL/BUY berlawanan dengan size besar?
    # Lebih baik kirim sinyal 'CLOSE_ALL' (Harus dihandle di Executor)

    # Cara Enterprise: Set State Global STOP
    await r_api.set("bot_state:global", "STOPPED")

    # Kirim instruksi langsung ke Channel AI (Output Fusion)
    # Executor akan membacanya sebagai instruksi final
    payload = {
        "final_action": "CLOSE_ALL",
        "symbol": "ALL",
        "reason": "EMERGENCY API CALL",
    }
    await r_api.publish(settings.CHANNEL_AI_ANALYSIS, json.dumps(payload))

    logger.warning("🚨 EMERGENCY CLOSE SIGNAL SENT!")
    return {"status": "executed", "action": "CLOSE_ALL"}


# 4. WEBSOCKET (Untuk Dashboard Realtime)
@app.websocket("/ws/live-feed")
async def ws_feed(websocket: WebSocket):
    await websocket.accept()
    if not r_api:
        await websocket.close()
        return

    ps = r_api.pubsub()
    await ps.subscribe(
        settings.CHANNEL_MARKET,  # Harga Live
        settings.CHANNEL_AI_ANALYSIS,  # Keputusan Final
        settings.CHANNEL_CONFIRMATION,  # Hasil Trade MT5
    )

    try:
        while True:
            msg = await ps.get_message(ignore_subscribe_messages=True, timeout=0.5)
            if msg:
                # Forward message dari Redis ke WebSocket Client
                await websocket.send_json(
                    {"channel": msg["channel"], "data": json.loads(msg["data"])}
                )
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"WS Error: {e}")
    finally:
        await ps.close()
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
