import asyncio
import json
import logging

import MetaTrader5 as mt5
import redis.asyncio as redis

from config import settings
from state_manager import state_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MT5-Executor")


class TradeExecutor:
    def __init__(self):
        self.r = None
        self.connected = False

    async def connect_redis(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    def _init_mt5_sync(self):
        """Fungsi sync MT5"""
        if not mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        ):
            logger.error(f"❌ MT5 Init Failed: {mt5.last_error()}")
            return False
        self.connected = True
        return True

    async def ensure_mt5(self):
        """Async wrapper untuk init MT5"""
        if not self.connected:
            return await asyncio.to_thread(self._init_mt5_sync)
        return True

    def place_order_sync(self, symbol, action, lots, sl=0.0, tp=0.0):
        # Pastikan connect (double check)
        if not self.connected:
            if not self._init_mt5_sync():
                return None

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 123456,
            "comment": "Forex AI Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Failed: {result.comment}")
            return None
        return result

    def close_all_positions_sync(self):
        """Emergency Close All (Sync)"""
        if not self._init_mt5_sync():
            return

        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                tick = mt5.symbol_info_tick(pos.symbol)
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": (
                        mt5.ORDER_TYPE_SELL
                        if pos.type == mt5.POSITION_TYPE_BUY
                        else mt5.ORDER_TYPE_BUY
                    ),
                    "position": pos.ticket,
                    "price": (
                        tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                    ),
                    "magic": 123456,
                    "comment": "EMERGENCY CLOSE",
                }
                mt5.order_send(req)
            logger.info("✅ All positions closed.")

    async def execute(self, s):
        act = s.get("final_action", "HOLD").upper()
        if act in ["HOLD", "HALTED", "READY"]:
            return

        if act == "CLOSE_ALL":
            await asyncio.to_thread(self.close_all_positions_sync)
            return

        current_mode = await state_manager.get_trading_mode()
        logger.info(f"⚡ Executing: {act} | Mode: {current_mode}")

        if current_mode == "LIVE":
            # Call MT5 in thread
            resp = await asyncio.to_thread(
                self.place_order_sync,
                s.get("symbol", settings.ACTIVE_SYMBOLS[0]),
                act,
                s.get("lots", settings.TRADE_LOTS),
                s.get("sl", 0.0),
                s.get("tp", 0.0),
            )

            if resp and self.r:
                await self.r.publish(
                    settings.CHANNEL_CONFIRMATION,
                    json.dumps(
                        {
                            "event": "ORDER_FILLED",
                            "order_id": resp.order,
                            "symbol": s.get("symbol"),
                            "action": act,
                            "price": resp.price,
                            "status": "OPEN",
                        }
                    ),
                )

    async def sync_mt5_state(self):
        """Membaca posisi dan akun dari MT5 secara Async"""
        if not await self.ensure_mt5():
            return

        # Jalankan di Thread agar tidak memblokir Loop Utama
        def _read_mt5():
            try:
                positions = mt5.positions_get()
                account = mt5.account_info()
                return positions, account
            except Exception as e:
                logger.error(f"MT5 Read Error: {e}")
                return None, None

        real_positions, account_info = await asyncio.to_thread(_read_mt5)

        # 1. Sync Account Info
        if account_info and self.r:
            pipe = self.r.pipeline()
            pipe.set("account_balance", float(account_info.balance))
            pipe.set("account_equity", float(account_info.equity))
            pipe.set("account_margin_free", float(account_info.margin_free))
            await pipe.execute()

        # 2. Sync Positions (Sederhana)
        # Di sini kita hanya mengecek jika posisi kosong di MT5 tapi ada di Redis -> Clear Redis
        if real_positions is not None:
            redis_pos = await state_manager.get_active_position()
            has_real_pos = len(real_positions) > 0

            # Jika MT5 kosong tapi Redis bilang ada -> Berarti sudah kena TP/SL
            if redis_pos and not has_real_pos:
                logger.info("⚠️ Posisi di MT5 hilang (Closed). Cleaning Redis...")
                await state_manager.clear_active_position()
                if self.r:
                    await self.r.publish(
                        settings.CHANNEL_CONFIRMATION,
                        json.dumps(
                            {"event": "ORDER_CLOSED_BY_BROKER", "status": "CLOSED"}
                        ),
                    )

    async def run(self):
        await self.connect_redis()
        await self.ensure_mt5()

        # Redis Listener
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_AI_ANALYSIS)
        logger.info("🎧 Executor Ready...")

        # Gunakan Task terpisah untuk loop sync agar tidak saling tunggu dengan listener
        asyncio.create_task(self.sync_loop_task())

        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    await self.execute(json.loads(msg["data"]))
                except Exception as e:
                    logger.error(f"Exec Error: {e}")

    async def sync_loop_task(self):
        while True:
            await self.sync_mt5_state()
            await asyncio.sleep(3)  # Sync tiap 3 detik


if __name__ == "__main__":
    asyncio.run(TradeExecutor().run())
