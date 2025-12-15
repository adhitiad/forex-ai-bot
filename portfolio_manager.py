import json
import logging

import redis.asyncio as redis

from config import settings
from logging_config import setup_logger

logger = setup_logger("PortfolioMgr")


class PortfolioManager:
    def __init__(self):
        self.r = None

    async def connect(self):
        if not self.r:
            self.r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )

    async def get_open_positions(self):
        """Hitung jumlah posisi aktif dari Redis"""
        await self.connect()
        keys = await self.r.keys("position:*")
        return len(keys)

    async def calculate_allocation(self, symbol, confidence):
        """
        Inti Logika Manajemen Risiko (Kelly Criterion Simplified)
        """
        await self.connect()

        # 1. Cek Max Positions
        open_count = await self.get_open_positions()
        if open_count >= settings.MAX_OPEN_POSITIONS:
            return (
                False,
                0.0,
                f"⛔ Max Positions ({open_count}/{settings.MAX_OPEN_POSITIONS})",
            )

        # 2. Ambil Equity (Simulasi atau Real)
        bal_str = await self.r.get("account_balance")
        balance = float(bal_str) if bal_str else 10000.0

        # 3. Hitung Risk Amount ($)
        # Risiko standar 1.5% dari equity per trade (setting di config)
        # Tapi kita pakai default 1% jika tidak ada di config
        risk_pct = getattr(settings, "RISK_PER_TRADE", 0.01)
        risk_amount = balance * risk_pct

        # 4. Sesuaikan dengan Confidence AI (Level 2 Logic)
        if confidence > 0.8:
            risk_amount *= 1.25  # Aggressive
        elif confidence < 0.6:
            risk_amount *= 0.5  # Defensive

        # 5. Konversi $ Risk ke Units (Lot)
        # Asumsi SL rata-rata 20 pips (0.0020) untuk pair Forex
        sl_distance = 0.0020
        units = int(risk_amount / sl_distance)

        # Min units Oanda = 1
        if units < 1:
            units = 100  # Micro lot minimum safety

        return True, units, f"✅ Alloc: {units} units (${risk_amount:.2f} risk)"

    async def register_position(self, symbol, data):
        await self.connect()
        await self.r.set(f"position:{symbol}", json.dumps(data))

    async def close_position(self, symbol):
        await self.connect()
        await self.r.delete(f"position:{symbol}")


# Singleton Instance
portfolio = PortfolioManager()  # Singleton Instance
