import asyncio
import json
import logging

import redis.asyncio as redis
from telegram import Bot

from config import settings
from logging_config import setup_logger

logger = setup_logger("Notifier")


class TelegramNotifier:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self.chat_id = settings.TELEGRAM_CHAT_ID

    async def send_msg(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            logger.error(f"Telegram Fail: {e}")

    async def run(self):
        logger.info("📢 Notifier Service Started...")
        await self.send_msg("🤖 Forex AI Bot: SYSTEM ONLINE (MT5 Connected)")

        ps = self.r.pubsub()
        # Dengarkan konfirmasi order dan pesan sistem
        await ps.subscribe(settings.CHANNEL_CONFIRMATION, settings.CHANNEL_SYSTEM)

        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    channel = msg["channel"]

                    text = ""
                    # 1. Notifikasi Trade (Order Filled / Closed)
                    if channel == settings.CHANNEL_CONFIRMATION:
                        event = data.get("event")
                        symbol = data.get("symbol")
                        action = data.get("action")

                        if event == "ORDER_FILLED":
                            text = f"🚀 <b>OPEN POSITION</b>\nSimbol: {symbol}\nAksi: {action}\nHarga: {data.get('price')}"
                        elif event == "ORDER_CLOSED_BY_BROKER":
                            text = f"🛑 <b>CLOSE (SL/TP)</b>\nSimbol: {symbol}\nStatus: {data.get('status')}"

                    # 2. Notifikasi Sistem (Training / Error)
                    elif channel == settings.CHANNEL_SYSTEM:
                        event = data.get("event")
                        if event == "TRAINING_COMPLETED":
                            text = "🧠 <b>AI Retraining Selesai</b>\nModel baru telah dimuat."
                        elif event == "ERROR":
                            text = f"⚠️ <b>SYSTEM ERROR</b>\n{data.get('message')}"

                    if text:
                        await self.send_msg(text)

                except Exception as e:
                    logger.error(f"Parse Error: {e}")


if __name__ == "__main__":
    asyncio.run(TelegramNotifier().run())
