import asyncio
import datetime

import redis.asyncio as redis

from config import settings
from logging_config import setup_logger

logger = setup_logger("MacroEngine")

# Simulasi Jadwal Event (Karena API ForexFactory berbayar/sulit scraping)
# Format waktu UTC
SCHEDULE = [
    {
        "day": "Friday",
        "time": "12:30",
        "event": "NFP (Non-Farm Payroll)",
        "impact": "HIGH",
    },
    {"day": "Wednesday", "time": "18:00", "event": "FOMC Statement", "impact": "HIGH"},
    {"day": "Thursday", "time": "12:30", "event": "CPI Data", "impact": "HIGH"},
]


class MacroEngine:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    async def check_calendar(self):
        """Cek apakah sekarang dekat dengan waktu berita High Impact"""
        now = datetime.datetime.now(datetime.timezone.utc)
        current_day = now.strftime("%A")
        current_time_str = now.strftime("%H:%M")  # Format HH:MM

        status = "SAFE"
        upcoming = "None"

        # Logika Sederhana: Cek manual list di atas
        for event in SCHEDULE:
            if event["day"] == current_day:
                # Cek jika waktu sekarang +/- 30 menit dari event
                event_dt = datetime.datetime.strptime(event["time"], "%H:%M").replace(
                    year=now.year,
                    month=now.month,
                    day=now.day,
                    tzinfo=datetime.timezone.utc,
                )

                # Selisih waktu dalam menit
                diff = (event_dt - now).total_seconds() / 60

                # Jika 30 menit sebelum atau 15 menit sesudah
                if -15 < diff < 30:
                    status = "DANGER"
                    upcoming = event["event"]
                    logger.warning(
                        f"🚨 MARKET DANGER ZONE: {upcoming} in {diff:.0f} mins"
                    )
                    break

        # Push Status ke Redis (Dibaca oleh Fusion Engine & LLM)
        await self.r.set("macro:status", status)
        await self.r.set("macro:next_event", upcoming)

        if status == "SAFE":
            logger.info(f"✅ Market Status: SAFE (No high impact events near)")

    async def run(self):
        logger.info("📅 Macro Engine Started (Economic Calendar)")
        while True:
            await self.check_calendar()
            await asyncio.sleep(60)  # Cek tiap menit


if __name__ == "__main__":
    asyncio.run(MacroEngine().run())
