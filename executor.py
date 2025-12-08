import asyncio
import json
import logging
import datetime
from typing import Optional
import ccxt.async_support as ccxt
import redis.asyncio as redis
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Executor")


class TradeExecutor:
    def __init__(self):
        self.r: Optional[redis.Redis] = None
        self.exchange: Optional[ccxt.Exchange] = None

    async def init_services(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        if not settings.TOKOCRYPTO_API_KEY:
            logger.warning("⚠️ PAPER MODE (No API Key)")
        else:
            try:
                self.exchange = ccxt.tokocrypto(
                    {
                        "apiKey": settings.TOKOCRYPTO_API_KEY,
                        "secret": settings.TOKOCRYPTO_SECRET,
                        "enableRateLimit": True,
                        "options": {"defaultType": "spot"},
                    }
                )
                await self.exchange.load_markets()
                logger.info("✅ Exchange Connected")
            except Exception as e:
                logger.error(f"❌ Exchange Error: {e}")

    async def check_min_notional(self, symbol, amount, price):
        if not self.exchange:
            return True
        try:
            market = self.exchange.market(symbol)
            cost = amount * price
            min_cost = market["limits"]["cost"]["min"] if market.get("limits") else 10.0
            if cost < min_cost:
                logger.warning(f"⛔ Rejected: Cost {cost} < Min {min_cost}")
                return False
            return True
        except:
            return True

    async def execute_trade(self, signal):
        action = signal.get("action")
        symbol = signal.get("symbol")
        price = float(signal.get("entry_price", 0))

        status = "FAILED"
        filled_price = price
        filled_amount = 0
        order_id = "SIM-" + str(int(datetime.datetime.now().timestamp()))

        if not self.exchange:
            logger.info(f"📝 [PAPER] {action} {symbol}")
            status = "SUCCESS"
        else:
            try:
                market = self.exchange.market(symbol)
                amount = 0
                order = None

                if action == "BUY":
                    bal = await self.exchange.fetch_balance()
                    quote = market["quote"]
                    free = float(bal[quote]["free"] or 0.0)
                    target = free * 0.98
                    amount = float(
                        self.exchange.amount_to_precision(symbol, target / price)
                    )

                    if await self.check_min_notional(symbol, amount, price):
                        logger.info("🚀 BUYING %s %s", amount, symbol)
                        order = await self.exchange.create_market_buy_order(
                            symbol, amount
                        )

                elif action == "SELL":
                    bal = await self.exchange.fetch_balance()
                    base = market["base"]
                    free = float(bal[base]["free"] or 0.0)
                    amount = float(self.exchange.amount_to_precision(symbol, free))

                    if await self.check_min_notional(symbol, amount, price):
                        logger.info("🚀 SELLING %s %s", amount, symbol)
                        order = await self.exchange.create_market_sell_order(
                            symbol, amount
                        )

                if order:
                    status = "SUCCESS"
                    order_id = order["id"]
                    filled_amount = order.get("filled", amount)
                    filled_price = order.get("average", price)
                    logger.info(f"✅ Filled: {order_id}")

            except Exception as e:
                logger.error(f"❌ Exec Error: {e}")

        if status == "SUCCESS":
            confirm = {
                "event": "ORDER_CONFIRMED",
                "original_action": action,
                "symbol": symbol,
                "filled_price": float(filled_price) if filled_price else price,
                "filled_amount": float(filled_amount),
                "order_id": str(order_id),
            }
            if self.r:
                await self.r.publish(settings.CHANNEL_CONFIRMATION, json.dumps(confirm))
                logger.info("📢 Confirmation Sent")

    async def run(self):
        await self.init_services()
        if not self.r:
            logger.error("❌ Redis not initialized")
            return
        pubsub = self.r.pubsub()
        await pubsub.subscribe(settings.CHANNEL_SIGNALS)
        logger.info("🎧 Waiting for Signals...")
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    if data.get("action") in ["BUY", "SELL", "CLOSE_BUY", "CLOSE_SELL"]:
                        act = data["action"]
                        if "CLOSE" in act:
                            act = "SELL" if "BUY" in act else "BUY"
                        data["action"] = act
                        await self.execute_trade(data)
                except Exception as e:
                    logger.error(e)


if __name__ == "__main__":
    asyncio.run(TradeExecutor().run())
