import asyncio
import json
import logging

import redis.asyncio as redis

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamManager")


class StreamManager:
    def __init__(self):
        self.r = None

    async def connect(self):
        try:
            kwargs = {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "decode_responses": True,
            }
            if settings.REDIS_PASSWORD:
                kwargs["password"] = settings.REDIS_PASSWORD
            self.r = redis.Redis(**kwargs)
            # Test connection
            await self.r.ping()
            logger.info("✅ Redis Connected")
        except Exception as e:
            logger.error(f"❌ Redis Error: {e}")

    async def push_market_data(self, symbol, data):
        if self.r:
            await self.r.xadd(settings.CHANNEL_MARKET, {"data": json.dumps(data)})
            await self.r.publish(settings.CHANNEL_MARKET, json.dumps(data))

    async def consume_market_data(self, group="brain_group", consumer="brain_1"):
        if not self.r:
            return []
        try:
            await self.r.xgroup_create(
                settings.CHANNEL_MARKET, group, id="0", mkstream=True
            )
        except:
            pass

        messages = await self.r.xreadgroup(
            group, consumer, {settings.CHANNEL_MARKET: ">"}, count=10
        )
        parsed = []
        ids = []
        for stream, entries in messages:
            for msg_id, content in entries:
                if "data" in content:
                    parsed.append(json.loads(content["data"]))
                    ids.append(msg_id)
        if ids:
            await self.r.xack(settings.CHANNEL_MARKET, group, *ids)
        return parsed

    async def push_signal(self, signal):
        if self.r:
            await self.r.publish(settings.CHANNEL_SIGNALS, json.dumps(signal))


streamor = StreamManager()
