import asyncio
import json
import logging
import os

import config
import redis
import torch
from fastapi import FastAPI
from features import fetcher, process_features
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from model import TimeSeriesTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Engine")
app = FastAPI()


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


@app.on_event("startup")
async def startup():
    """Start the trading loop."""
    asyncio.create_task(trading_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
