import asyncio
import json
import logging
import os
import time
from typing import Optional

import config
import redis
import torch
from fastapi import FastAPI, HTTPException
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
    symbol: Optional[str] = None  # Optional, defaults to config


class BacktestRequest(BaseModel):
    symbol: Optional[str] = None
    days: int = 60


class TrainRequest(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None


class HealthResponse(BaseModel):
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
            full_df, features = process_features(raw_df)
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


# Global variable to track trading status
trading_active = False


@app.on_event("startup")
async def startup():
    """Start the trading loop."""
    global trading_active
    trading_active = True
    asyncio.create_task(trading_loop())


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
        trading_active=trading_active,
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
        "trading_active": trading_active,
        "config": {
            "timeframe": config.TIMEFRAME,
            "take_profit_pct": config.TAKE_PROFIT_PCT,
            "stop_loss_pct": config.STOP_LOSS_PCT,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "lot_size": config.LOT_SIZE,
        },
        "timestamp": str(asyncio.get_event_loop().time()),
    }


@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    """Get AI prediction for current market data."""
    try:
        symbol = request.symbol or SYMBOL

        # Fetch latest data
        raw_df = await fetcher.fetch_market_data(
            symbol=symbol, period="60d", interval=config.TIMEFRAME
        )

        if raw_df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for symbol {symbol}"
            )

        # Process features
        full_df, features = process_features(raw_df)
        if len(features) < config.SEQ_LEN:
            raise HTTPException(
                status_code=400, detail="Insufficient data for prediction"
            )

        # Make prediction
        tensor_in = torch.FloatTensor(features[-config.SEQ_LEN :]).unsqueeze(0)
        with torch.no_grad():
            probs = model(tensor_in).numpy()[0]  # [Buy, Sell, Hold]

        current_price = float(raw_df.iloc[-1]["close"])

        # Determine action
        action = "HOLD"
        confidence = float(max(probs))
        if probs[0] > config.CONFIDENCE_THRESHOLD:
            action = "BUY"
        elif probs[1] > config.CONFIDENCE_THRESHOLD:
            action = "SELL"

        return {
            "symbol": symbol,
            "current_price": current_price,
            "prediction": {
                "action": action,
                "confidence": confidence,
                "probabilities": {
                    "buy": float(probs[0]),
                    "sell": float(probs[1]),
                    "hold": float(probs[2]),
                },
            },
            "timestamp": str(raw_df.index[-1]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        test_signal = {
            "symbol": SYMBOL_BINANCE,
            "action": "BUY",
            "price": 50000.0,
            "confidence": 0.75,
            "timestamp": str(asyncio.get_event_loop().time()),
            "test": True,
        }

        # Publish to Redis
        r.publish("trade_signals", json.dumps(test_signal))

        return {"message": "Test signal sent", "signal": test_signal}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
