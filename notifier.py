import asyncio
import json
import logging
import os
from datetime import datetime

import redis.asyncio as redis
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from config import settings

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Notifier")


class TelegramNotifier:
    def __init__(self):
        self.r = None
        self.bot = None
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.token = settings.TELEGRAM_BOT_TOKEN

    async def connect_redis(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            db=settings.REDIS_DB,  # Support Multi-Pair DB
        )

    async def send_message(self, text):
        if not self.token or not self.chat_id:
            return
        if not self.bot:
            self.bot = Bot(token=self.token)
        try:
            await self.bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"❌ Gagal kirim Telegram: {e}")

    def format_signal_message(self, data):
        """Format pesan Sinyal AI (Prediksi)"""
        action = data.get("action", "UNKNOWN")
        symbol = data.get("symbol", settings.ACTIVE_SYMBOLS[0])
        price = float(data.get("entry_price", 0))
        tp = float(data.get("tp", 0))
        sl = float(data.get("sl", 0))
        confidence = float(data.get("confidence", 0))
        reason = data.get("reason", "-")
        timestamp = datetime.now().strftime("%H:%M:%S")

        icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"

        # Format angka desimal (5 digit utk forex, 2 utk Gold/Yen)
        fmt = ".2f" if "JPY" in symbol or "XAU" in symbol else ".5f"

        msg = (
            f"{icon} <b>AI SIGNAL ALERT</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Action:</b> {action}\n"
            f"<b>Entry:</b> {price:{fmt}}\n"
            f"🎯 <b>TP:</b> {tp:{fmt}}\n"
            f"🛡️ <b>SL:</b> {sl:{fmt}}\n\n"
            f"<b>Confidence:</b> {confidence:.2f}%\n"
            f"<b>Time:</b> {timestamp}\n"
            f"🧠 <b>Reason:</b> {reason}"
        )
        return msg

    def format_execution_message(self, data):
        """Format pesan Eksekusi Order (Real Trade)"""
        event = data.get("event", "UNKNOWN")
        symbol = data.get("symbol", "-")
        action = data.get("action", "-")
        price = float(data.get("price", 0))
        tp = float(data.get("tp", 0))
        sl = float(data.get("sl", 0))
        status = data.get("status", "UNKNOWN")

        fmt = ".2f" if "JPY" in symbol or "XAU" in symbol else ".5f"

        if event == "MANUAL_TRADE_REQUIRED":
            return (
                f"🚨 <b>STOCKBIT SIGNAL ALERT!</b>\n\n"
                f"📈 <b>Saham:</b> {symbol}\n"
                f"👉 <b>Action:</b> {action}\n"
                f"💵 <b>Price:</b> {price:,.0f}\n\n"
                f"⚠️ <i>Bot tidak bisa auto-trade di Stockbit.</i>\n"
                f"<b>Segera buka aplikasi Stockbit dan entry manual!</b>"
            )
        if event == "ORDER_FILLED":
            return (
                f"⚡ <b>ORDER EXECUTED</b>\n\n"
                f"<b>{symbol}</b> - {action}\n"
                f"<b>Open Price:</b> {price:{fmt}}\n"
                f"🎯 <b>TP:</b> {tp:{fmt}}\n"
                f"🛡️ <b>SL:</b> {sl:{fmt}}\n\n"
                f"Status: ✅ {status}"
            )
        elif event == "ORDER_CLOSED_BY_BROKER":
            pnl = float(data.get("pnl", 0)) if "pnl" in data else 0.0
            emoji = "💰" if pnl >= 0 else "💸"
            return (
                f"{emoji} <b>POSITION CLOSED</b>\n\n"
                f"<b>{symbol}</b>\n"
                f"<b>PnL:</b> ${pnl:.2f}\n"
                f"Closed by Broker (TP/SL)"
            )

        return None

    async def run(self):
        logger.info("📲 Telegram Notifier Started (With TP/SL)...")
        await self.connect_redis()

        # Ensure Redis connection is established
        if not self.r:
            logger.error("❌ Redis connection failed, cannot start notifier")
            return

        ps = self.r.pubsub()
        await ps.subscribe(
            settings.CHANNEL_AI_ANALYSIS,
            settings.CHANNEL_CONFIRMATION,
            settings.CHANNEL_SYSTEM,
        )

        async for msg in ps.listen():
            if msg["type"] == "message":
                try:
                    channel = msg["channel"]
                    data = json.loads(msg["data"])
                    text_to_send = None

                    if channel == settings.CHANNEL_AI_ANALYSIS:
                        if data.get("action") in ["BUY", "SELL"]:
                            text_to_send = self.format_signal_message(data)

                    elif channel == settings.CHANNEL_CONFIRMATION:
                        text_to_send = self.format_execution_message(data)

                    elif channel == settings.CHANNEL_SYSTEM:
                        if data.get("event") in ["START", "STOP"]:
                            text_to_send = f"🖥️ <b>SYSTEM:</b> {data.get('event')}"

                    if text_to_send:
                        await self.send_message(text_to_send)

                except Exception as e:
                    logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(TelegramNotifier().run())
