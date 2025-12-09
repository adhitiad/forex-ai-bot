import asyncio
import datetime
import json
import logging

import redis.asyncio as redis

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PaperTrade")


class PaperExecutor:
    def __init__(self):
        self.r = None

    async def init_redis(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    async def execute_trade(self, signal):
        action = signal.get("action")
        try:
            entry_price = float(signal.get("entry_price", 0))
        except:
            return

        # 1. Tentukan Nilai Pip (Pip Value)
        # JPY pairs (e.g. USDJPY) 1 pip = 0.01, Lainnya (EURUSD) = 0.0001
        is_jpy = "JPY" in settings.ACTIVE_SYMBOL or "jpy" in settings.ACTIVE_SYMBOL
        pip_unit = 0.01 if is_jpy else 0.0001

        # 2. Hitung Jarak TP dan SL
        tp_dist = settings.TAKE_PROFIT_PIPS * pip_unit
        sl_dist = settings.STOP_LOSS_PIPS * pip_unit

        # 3. Hitung Harga TP dan SL
        if action == "BUY":
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
            color = "🟢"  # Hijau untuk Buy
        else:  # SELL
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist
            color = "🔴"  # Merah untuk Sell

        # 4. TAMPILKAN DATA LENGKAP (Display)
        print("\n" + "=" * 50)
        print(f"{color} SIGNAL RECEIVED: {action} {settings.ACTIVE_SYMBOL}")
        print(f"   💵 Entry Price : {entry_price:.5f}")
        print(f"   🎯 Take Profit : {tp_price:.5f} (+{settings.TAKE_PROFIT_PIPS} pips)")
        print(f"   🛡️ Stop Loss   : {sl_price:.5f} (-{settings.STOP_LOSS_PIPS} pips)")
        print(f"   📦 Units       : {settings.TRADE_UNITS}")
        print("=" * 50 + "\n")

        # 5. Kirim Konfirmasi ke Brain (Simulasi Order Filled)
        order_id = f"SIM-{int(datetime.datetime.now().timestamp())}"
        confirm = {
            "event": "ORDER_CONFIRMED",
            "original_action": action,
            "symbol": settings.ACTIVE_SYMBOL,
            "filled_price": entry_price,
            "filled_amount": settings.TRADE_UNITS,
            "order_id": order_id,
            "status": "FILLED_SIMULATED",
        }

        if self.r:
            await self.r.publish(settings.CHANNEL_CONFIRMATION, json.dumps(confirm))
            logger.info(f"✅ Order {order_id} Confirmed to System")

    async def run(self):
        await self.init_redis()
        if not self.r:
            logger.error("❌ Redis connection failed")
            return

        pubsub = self.r.pubsub()
        await pubsub.subscribe(settings.CHANNEL_SIGNALS)
        logger.info("🎧 Paper Executor Ready & Waiting for Signals...")

        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    if data.get("action") in ["BUY", "SELL"]:
                        await self.execute_trade(data)
                except Exception as e:
                    logger.error(f"Error processing signal: {e}")


if __name__ == "__main__":
    asyncio.run(PaperExecutor().run())
