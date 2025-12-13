import asyncio
import datetime
import json
import logging
from typing import cast

import oandapyV20
import oandapyV20.endpoints.orders as orders
import pandas as pd
import redis.asyncio as redis
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    StopLossDetails,
    TakeProfitDetails,
)

from config import settings
from state_manager import state_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealExecutor")


class TradeExecutor:
    def __init__(self):
        self.r = None
        self.client = None
        # OANDA Init akan dilakukan dinamis saat execute jika perlu

    async def connect(self):
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r = redis.Redis(**kwargs)

    def place_oanda_order(self, action, units, tp, sl):
        try:
            client = oandapyV20.API(
                access_token=settings.OANDA_ACCESS_TOKEN, environment=settings.OANDA_ENV
            )
            qty = units if "BUY" in action else -units
            cfg = MarketOrderRequest(
                instrument=settings.OANDA_SYMBOL,
                units=qty,
                takeProfitOnFill=TakeProfitDetails(price=str(round(tp, 5))).data,
                stopLossOnFill=StopLossDetails(price=str(round(sl, 5))).data,
            )
            r = orders.OrderCreate(settings.OANDA_ACCOUNT_ID, data=cfg.data)
            return client.request(r)
        except Exception as e:
            logger.error(f"OANDA Error: {e}")
            return None

    async def execute(self, s):
        act = s.get("final_action", "HOLD").upper()
        if act in ["HOLD", "HALTED", "READY"] or s.get("entry_price", 0) == 0:
            return

        current_mode = await state_manager.get_trading_mode()
        logger.info(f"⚡ Execution Attempt: {act} in [{current_mode}]")

        broker_id = "PAPER-SIM"
        if current_mode == "LIVE":
            if settings.OANDA_ACCESS_TOKEN:
                resp = self.place_oanda_order(
                    act, settings.TRADE_UNITS, s.get("tp"), s.get("sl")
                )
                if resp and isinstance(resp, dict):
                    broker_id = f"OANDA-{resp['orderFillTransaction']['id']}"
            else:
                logger.error("Missing OANDA Credentials!")
                return

        print(
            f"\n{'='*40}\nEXECUTING {act}\nPrice: {s.get('entry_price')}\nTP: {s.get('tp')} | SL: {s.get('sl')}\nMode: {current_mode}\n{'='*40}\n"
        )

        if self.r:
            side = "BUY" if "BUY" in act else "SELL"
            await self.r.publish(
                settings.CHANNEL_CONFIRMATION,
                json.dumps(
                    {
                        "event": "ORDER_FILLED",
                        "order_id": broker_id,
                        "symbol": settings.ACTIVE_SYMBOL,
                        "action": side,
                        "open_price": s.get("entry_price"),
                        "tp": s.get("tp"),
                        "sl": s.get("sl"),
                        "status": "OPEN",
                        "mode": current_mode,
                    }
                ),
            )

    async def run(self):
        await self.connect()
        self.r = cast(redis.Redis, self.r)
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_AI_ANALYSIS)
        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    await self.execute(json.loads(msg["data"]))
                except:
                    pass


if __name__ == "__main__":
    asyncio.run(TradeExecutor().run())
