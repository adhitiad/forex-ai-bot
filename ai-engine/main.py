"""
AI Engine - Main Application
"""

import asyncio
import json
import logging
import os
from typing import List, Optional

import config
import redis
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from features import fetcher, process_features, train_model_async
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from model import TimeSeriesTransformer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Engine")
app = FastAPI(
    title="Forex AI Bot - AI Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Pydantic Models
class PredictionRequest(BaseModel):
    """
    Request model for making predictions.
    """

    symbol: Optional[str] = None  # Optional, defaults to config


class BacktestRequest(BaseModel):
    """
    Request model for backtesting.
    """

    symbol: Optional[str] = None
    days: int = 60


class TrainRequest(BaseModel):
    """
    Request model for training.
    """

    symbol: Optional[str] = None
    timeframe: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Response model for health checks.
    """

    status: str
    timestamp: str
    model_loaded: bool
    trading_active: bool


r = redis.Redis(
    host="redis-16018.c334.asia-southeast2-1.gce.cloud.redislabs.com",
    port=16018,
    decode_responses=True,
    username="default",
    password="fNzirXEiYdNFVFA3tbHGPoTJA00q2jX9",
)

# InfluxDB
influx = InfluxDBClient(
    url="https://us-east-1-1.aws.cloud2.influxdata.com",
    token="x7GHjgNhOOWdosiY2eu-Z5kZS4PceGIYq3eOZfdYbJy5sbunIk3M2ZwQSh04SRBBFZB67qoZNfLdhKO9afcEpw==",
    org="hjgf",
)
write_api = influx.write_api(write_options=SYNCHRONOUS)

# Model
model = TimeSeriesTransformer(input_dim=4, d_model=128)
if os.path.exists("/data/trained_model.pth"):
    model.load_state_dict(torch.load("/data/trained_model.pth"))
model.eval()

SYMBOL = config.SYMBOL_YF
SYMBOL_BINANCE = "BTCUSDT"


# WebSocket Manager for Real-Time Mode
class WebSocketManager:
    """
    Class to manage WebSocket connections.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket client connected. Total clients: %s",
            len(self.active_connections),
        )

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                "WebSocket client disconnected. Total clients: %s",
                len(self.active_connections),
            )

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except (WebSocketDisconnect, OSError, ConnectionError) as e:
                logger.error("Error sending to WebSocket client: %s", e)
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)


# Global WebSocket Manager
ws_manager = WebSocketManager()


class RealtimeState:
    """
    Class to manage the real-time mode state.
    """

    def __init__(self):
        self.active = False


realtime_state = RealtimeState()


class TradingState:
    """
    Class to manage the trading state.
    """

    def __init__(self):
        self.active = False


trading_state = TradingState()


@app.get("/trade/start")
async def trading_loop():
    """
    Trading loop for executing trades based on model predictions.
    """
    logger.info("🚀 AI Engine Active on %s", SYMBOL)
    logger.info(
        "🎯 Target TP: %s%% | SL: %s%%",
        config.TAKE_PROFIT_PCT * 100,
        config.STOP_LOSS_PCT * 100,
    )

    while True:
        try:
            # Menggunakan Config
            raw_df = await fetcher.fetch_market_data(
                symbol=SYMBOL, period="2y", interval=config.TIMEFRAME
            )

            if raw_df.empty:
                await asyncio.sleep(10)
                continue

            # 2. Process
            _, features = process_features(raw_df)
            if len(features) < 30:
                continue

            tensor_in = torch.FloatTensor(features[-30:]).unsqueeze(0)

            # 3. Predict
            with torch.no_grad():
                probs = model(tensor_in).numpy()[0]  # [Buy, Sell, Hold]

            # 4. Logic
            action = "HOLD"
            if probs[0] > 0.6:
                action = "BUY"
            elif probs[1] > 0.6:
                action = "SELL"

            price = float(raw_df.iloc[-1]["close"])

            # Broadcast real-time data to WebSocket clients
            if realtime_state.active:
                realtime_data = {
                    "type": "market_data",
                    "symbol": SYMBOL,
                    "price": price,
                    "timestamp": str(raw_df.index[-1]),
                    "prediction": {
                        "action": action,
                        "probabilities": {
                            "buy": float(probs[0]),
                            "sell": float(probs[1]),
                            "hold": float(probs[2]),
                        },
                        "confidence": float(max(probs)),
                    },
                    "features": {
                        "rsi": float(features[-1][1]) if len(features) > 0 else None,
                        "ema": float(features[-1][2]) if len(features) > 0 else None,
                        "atr": float(features[-1][3]) if len(features) > 0 else None,
                    },
                }
                await ws_manager.broadcast(realtime_data)

            # Logic Kirim Sinyal ke Go
            if action != "HOLD":
                signal = {
                    "symbol": SYMBOL,  # XAUUSD=X
                    "action": action,
                    "price": price,
                    # Kirim juga parameter TP/SL ke Go Executor agar Go yang pasang limit order
                    "tp_price": (
                        price * (1 + config.TAKE_PROFIT_PCT)
                        if action == "BUY"
                        else price * (1 - config.TAKE_PROFIT_PCT)
                    ),
                    "sl_price": (
                        price * (1 - config.STOP_LOSS_PCT)
                        if action == "BUY"
                        else price * (1 + config.STOP_LOSS_PCT)
                    ),
                    "lot": config.LOT_SIZE,
                }
                r.publish("trade_signals", json.dumps(signal))

                # Broadcast signal to WebSocket clients
                if realtime_state.active:
                    signal_broadcast = {
                        "type": "trade_signal",
                        "signal": signal,
                        "timestamp": str(asyncio.get_event_loop().time()),
                    }
                    await ws_manager.broadcast(signal_broadcast)

            # 5. Save & Publish
            p = (
                Point("market_tick")
                .tag("symbol", SYMBOL_BINANCE)
                .field("price", price)
                .field("prob_buy", float(probs[0]))
            )
            write_api.write(bucket="ADH", record=p)

            if action != "HOLD":
                signal = {
                    "symbol": SYMBOL_BINANCE,
                    "action": action,
                    "price": price,
                    "confidence": float(max(probs)),
                    "timestamp": str(raw_df.index[-1]),
                }
                r.publish("trade_signals", json.dumps(signal))
                logger.info("🔥 SIGNAL: %s", signal)

        except (ValueError, RuntimeError, KeyError, IOError) as e:
            logger.error("Trading loop error: %s", e)
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup():
    """Start the trading loop."""
    trading_state.active = True
    asyncio.create_task(trading_loop())


# WebSocket Endpoints
@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """WebSocket endpoint for real-time AI engine data."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, clients can send messages if needed
            data = await websocket.receive_text()
            # For now, just echo or handle simple commands
            if data == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": str(asyncio.get_event_loop().time())}
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = os.path.exists(config.MODEL_FILE) or os.path.exists(
        "/data/trained_model.pth"
    )
    return HealthResponse(
        status="healthy",
        timestamp=str(asyncio.get_event_loop().time()),
        model_loaded=model_loaded,
        trading_active=trading_state.active,
    )


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train the AI model."""
    try:
        # Extract parameters from the request
        symbol = request.symbol or SYMBOL
        timeframe = request.timeframe or config.TIMEFRAME

        # Start training process (this should be asynchronous)
        asyncio.create_task(train_model_async(symbol, timeframe))

        return {"message": "Training started", "symbol": symbol, "timeframe": timeframe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/status")
async def get_status():
    """Get current AI engine status."""
    model_loaded = os.path.exists(config.MODEL_FILE) or os.path.exists(
        "/data/trained_model.pth"
    )
    return {
        "symbol": SYMBOL,
        "model_loaded": model_loaded,
        "trading_active": trading_state.active,
        "realtime_mode_active": realtime_state.active,
        "websocket_clients": len(ws_manager.active_connections),
        "config": {
            "timeframe": config.TIMEFRAME,
            "take_profit_pct": config.TAKE_PROFIT_PCT,
            "stop_loss_pct": config.STOP_LOSS_PCT,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "lot_size": config.LOT_SIZE,
        },
        "timestamp": str(asyncio.get_event_loop().time()),
    }


@app.post("/realtime/enable")
async def enable_realtime_mode():
    """Enable real-time mode for WebSocket broadcasting."""
    realtime_state.active = True
    logger.info("🔄 Real-time mode enabled")
    return {"message": "Real-time mode enabled"}


@app.post("/backtest")
async def run_backtest_endpoint(request: BacktestRequest):
    """Run backtest for given parameters."""
    try:
        # Note: This is a simplified version. In production, you'd want to run this asynchronously
        # and return a job ID for status tracking

        symbol = request.symbol or config.SYMBOL_YF
        days = request.days

        # This would ideally be run in background and return job status
        # For now, return placeholder
        return {
            "message": f"Backtest scheduled for {symbol} over {days} days",
            "status": "scheduled",
            "job_id": f"backtest_{symbol}_{days}_{asyncio.get_event_loop().time()}",
            "note": "Backtest execution not implemented in API yet - use backtest.py directly",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    model_path = (
        config.MODEL_FILE
        if os.path.exists(config.MODEL_FILE)
        else "/data/trained_model.pth"
    )
    model_exists = os.path.exists(model_path)

    return {
        "model_exists": model_exists,
        "model_path": model_path,
        "input_dim": 4,
        "seq_len": config.SEQ_LEN,
        "features": ["close", "RSI_14", "EMA_20", "ATRr_14"],
        "output_classes": ["BUY", "SELL", "HOLD"],
        "last_modified": str(os.path.getmtime(model_path)) if model_exists else None,
    }


@app.post("/signal/test")
async def test_signal():
    """Test signal generation (development endpoint)."""
    try:
        # Generate a test signal
        signal = {
            "symbol": SYMBOL_BINANCE,
            "action": "BUY",
            "price": 50000.0,
            "confidence": 0.75,
            "timestamp": str(asyncio.get_event_loop().time()),
            "test": True,
        }

        # Publish to Redis
        r.publish("trade_signals", json.dumps(signal))

        return {"message": "Test signal sent", "signal": signal}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
