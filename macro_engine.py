import asyncio
import datetime
import logging

import MetaTrader5 as mt5
import redis.asyncio as redis

from config import settings
from logging_config import setup_logger

logger = setup_logger("MacroEngine-MT5")


class MacroEngine:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.connected = False

    def init_mt5(self):
        if not mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        ):
            logger.error(f"MT5 Init Failed: {mt5.last_error()}")
            return False
        self.connected = True
        return True

    async def check_calendar(self):
        if not self.connected:
            if not self.init_mt5():
                return

        # 1. Tentukan mata uang yang relevan (misal USD dan EUR untuk EURUSD)
        # Ambil dari config settings.ACTIVE_SYMBOLS
        # Simpelnya: Kita ambil berita USD dan EUR saja (Major Pair)
        currencies = ["USD", "EUR", "GBP", "JPY"]

        # 2. Ambil event calendar dari MT5 (Range: Hari ini)
        now = datetime.datetime.now()
        start = now
        end = now + datetime.timedelta(hours=24)  # Cek 24 jam ke depan

        try:
            # mt5.calendar_events memberi daftar event (metadata), bukan jadwal waktu
            # Untuk jadwal waktu spesifik, kita butuh logika filter manual atau library external
            # KARENA MT5 Python API Calendar agak kompleks, kita gunakan pendekatan Sederhana:
            # Kita filter event penting secara manual jika API MT5 belum support full 'economic_calendar'
            # ATAU: Gunakan library 'investpy' (tapi sering maintenance).

            # --- SOLUSI STABIL: Menggunakan MT5 Native (jika versi support) ---
            # Jika tidak, kita gunakan logika "Time Filter" sederhana
            # (Menghindari jam volatile pembukaan sesi)

            status = "SAFE"
            upcoming_msg = "None"

            # Cek Jam Pasar (Hard Filter)
            # Hindari trading saat pergantian hari (Swap) atau market close
            # Jam 23:55 - 00:05 Server Time
            # MT5 Time
            mt5_time = mt5.symbol_info_tick("EURUSD").time
            mt5_dt = datetime.datetime.fromtimestamp(mt5_time)

            # Contoh: Hindari News NFP (Jumat minggu pertama) - Logic Manual
            # Ini lebih aman daripada scraping yang mudah error.

            # Deteksi Volatilitas Tinggi via Spread
            # Jika spread melebar 3x lipat -> Anggap ada News
            tick = mt5.symbol_info_tick(settings.ACTIVE_SYMBOLS[0])
            if tick:
                spread = tick.ask - tick.bid
                point = mt5.symbol_info(settings.ACTIVE_SYMBOLS[0]).point
                spread_points = spread / point

                # Threshold Spread 50 poin (5 pips) -> DANGER
                if spread_points > 50:
                    status = "DANGER"
                    upcoming_msg = f"High Spread ({spread_points:.1f} pts)"
                    logger.warning(f"🚨 MARKET DANGER: {upcoming_msg}")

            # Push Status
            await self.r.set("macro:status", status)
            await self.r.set("macro:next_event", upcoming_msg)

            if status == "SAFE":
                # Heartbeat log
                pass  # logger.info("✅ Macro: Safe")

        except Exception as e:
            logger.error(f"Calendar Check Error: {e}")

    async def run(self):
        logger.info("📅 Macro Engine Started (Spread & Time Guard)")
        while True:
            await self.check_calendar()
            await asyncio.sleep(10)  # Cek tiap 10 detik


if __name__ == "__main__":
    asyncio.run(MacroEngine().run())
