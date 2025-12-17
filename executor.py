# File: executor.py
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

    def init_mt5(self):
        """Koneksi ke Terminal FBS MT5"""
        if not mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        ):
            logger.error(f"❌ MT5 Init Failed: {mt5.last_error()}")
            return False

        self.connected = True
        logger.info(f"✅ Connected to FBS MT5 Account: {settings.MT5_LOGIN}")
        return True

    def place_order_sync(self, symbol, action, lots, sl=0.0, tp=0.0):
        """Fungsi sinkronus untuk order MT5"""
        if not self.connected:
            if not self.init_mt5():
                return None

        # Cek harga market
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"❌ Symbol {symbol} not found")
            return None

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action == "BUY" else tick.bid

        # Request structure MT5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 123456,  # Magic Number Bot
            "comment": "Forex AI Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Kirim Order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order Failed: {result.comment} ({result.retcode})")
            return None

        return result

    async def execute(self, s):
        act = s.get("final_action", "HOLD").upper()
        if act in ["HOLD", "HALTED", "READY"]:
            return

        current_mode = await state_manager.get_trading_mode()
        logger.info(f"⚡ Signal Received: {act} | Mode: {current_mode}")

        if current_mode == "LIVE":
            # Jalankan fungsi blocking MT5 di thread terpisah agar tidak lag
            resp = await asyncio.to_thread(
                self.place_order_sync,
                settings.ACTIVE_SYMBOLS[0],  # Atau ambil dari s['symbol']
                "BUY" if "BUY" in act else "SELL",
                settings.TRADE_LOTS,
                s.get("sl", 0.0),
                s.get("tp", 0.0),
            )

            if resp:
                logger.info(
                    f"✅ OPEN POS: #{resp.order} {act} {resp.volume} Lots @ {resp.price}"
                )

                # Publish konfirmasi ke Redis
                if self.r:
                    await self.r.publish(
                        settings.CHANNEL_CONFIRMATION,
                        json.dumps(
                            {
                                "event": "ORDER_FILLED",
                                "order_id": resp.order,
                                "symbol": settings.ACTIVE_SYMBOLS[0],
                                "action": act,
                                "price": resp.price,
                                "status": "OPEN",
                            }
                        ),
                    )
        act = s.get("final_action", "HOLD").upper()

        # === TAMBAHKAN BLOK INI ===
        if act == "CLOSE_ALL":
            logger.warning("🚨 EXECUTING EMERGENCY CLOSE ALL!")
            if self.init_mt5():
                # Loop semua posisi aktif
                positions = mt5.positions_get()
                if positions:
                    for pos in positions:
                        # Close manual
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
                                mt5.symbol_info_tick(pos.symbol).bid
                                if pos.type == mt5.POSITION_TYPE_BUY
                                else mt5.symbol_info_tick(pos.symbol).ask
                            ),
                            "magic": 123456,
                            "comment": "EMERGENCY CLOSE",
                        }
                        mt5.order_send(req)
                    logger.info("✅ All positions closed.")
            return
        # ==========================

    async def sync_positions(self):
        """Mencocokkan data Redis dengan Realita di MT5"""
        if not self.connected:
            self.init_mt5()

        # 1. Ambil posisi asli dari MT5
        real_positions = mt5.positions_get()
        if real_positions is None:
            return

        real_symbols = [p.symbol for p in real_positions]

        # 2. Cek status di Redis
        # Asumsi kita trading 1 simbol aktif dulu sesuai config
        active_symbol = settings.ACTIVE_SYMBOLS[0]

        # Ambil data posisi tersimpan di Redis
        redis_pos = await state_manager.get_active_position()  # Return dict atau None

        # KASUS A: Redis bilang OPEN, tapi di MT5 Kosong (Berarti kena SL/TP saat offline)
        if redis_pos and (active_symbol not in real_symbols):
            logger.warning(
                f"⚠️ GHOST POSITION DETECTED! Clearing Redis state for {active_symbol}..."
            )
            await state_manager.clear_active_position()

            # Opsional: Kirim notif bahwa trade sudah close
            await self.r.publish(
                settings.CHANNEL_CONFIRMATION,
                json.dumps(
                    {
                        "event": "ORDER_CLOSED_BY_BROKER",
                        "symbol": active_symbol,
                        "status": "CLOSED",
                    }
                ),
            )

        # KASUS B: Redis Kosong, tapi di MT5 ada Posisi (Manual trade / Error save)
        elif not redis_pos and (active_symbol in real_symbols):
            # Kita anggap ini trade manual user, biarkan atau catat ke Redis
            pos = [p for p in real_positions if p.symbol == active_symbol][0]
            logger.info(f"👀 Found existing trade on MT5: {pos.ticket}")
            await state_manager.set_active_position(
                active_symbol,
                "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                pos.price_open,
                pos.volume,
                pos.tp,
                pos.sl,
            )

    # Update fungsi run() untuk menjalankan sync loop
    async def run(self):
        await self.connect_redis()
        if self.init_mt5():
            # Jalankan Sync Loop di background
            asyncio.create_task(self.sync_loop())

            ps = self.r.pubsub()
            await ps.subscribe(settings.CHANNEL_AI_ANALYSIS)
            logger.info("🎧 Executor Ready & Listening...")

            async for msg in ps.listen():
                if msg["type"] == "message":
                    try:
                        await self.execute(json.loads(msg["data"]))
                    except Exception as e:
                        logger.error(f"Exec Error: {e}")

    async def sync_account_info(self):
        """Ambil Info Akun (Saldo, Equity) dari MT5"""
        if not self.connected:
            return

        account_info = mt5.account_info()
        if account_info is None:
            return

        # Simpan ke Redis agar Portfolio Manager membacanya
        # Kita simpan balance dan equity
        pipeline = self.r.pipeline()
        pipeline.set("account_balance", float(account_info.balance))
        pipeline.set("account_equity", float(account_info.equity))
        pipeline.set("account_margin_free", float(account_info.margin_free))
        await pipeline.execute()

        # Log warning jika margin kritis
        if account_info.margin_level < 100 and account_info.margin_level > 0:
            logger.warning(f"⚠️ LOW MARGIN: {account_info.margin_level}%")

    async def sync_loop(self):
        """Looping tugas rutin: Cek Posisi & Cek Saldo"""
        while True:
            try:
                # 1. Cek Posisi (Sync Order)
                await self.sync_positions()

                # 2. Cek Saldo (Agar Risk Management Valid)
                await self.sync_account_info()

            except Exception as e:
                logger.error(f"Sync Loop Error: {e}")

            # Update tiap 5 detik
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(TradeExecutor().run())
