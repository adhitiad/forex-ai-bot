import asyncio
import json
import logging

import redis.asyncio as redis

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamManager:
    def __init__(self):
        self.r = None
        self.STREAM_MARKET = "market_data"
        self.CHANNEL_SIGNALS = "channel_signals"
        self.CHANNEL_CONTROL = "channel_control"
        self.connect()

    def connect(self):
        try:
            self.r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Redis Error: {e}")

    async def push_market_data(self, symbol, data):
        if self.r:
            await self.r.xadd(self.STREAM_MARKET, {"data": json.dumps(data)})
            await self.r.publish("channel_market", json.dumps(data))
            logger.info(f"Pushed market data: {symbol}")

    async def consume_market_data(
        self, group_name="brain_group", consumer_name="brain_1"
    ):
        if not self.r:
            return []
        try:
            await self.r.xgroup_create(
                self.STREAM_MARKET, group_name, id="0", mkstream=True
            )
            logger.info(f"Created consumer group: {group_name}")
        except Exception as e:
            logger.error(f"Error creating consumer group: {e}")
            return []

        messages = await self.r.xreadgroup(
            group_name, consumer_name, {self.STREAM_MARKET: ">"}, count=10
        )
        parsed = []
        for stream, entries in messages:
            for message_id, content in entries:
                if "data" in content:
                    parsed.append(json.loads(content["data"]))
                    await self.r.xack(self.STREAM_MARKET, group_name, message_id)
        return parsed

    async def push_signal(self, signal):
        if self.r:
            await self.r.publish(self.CHANNEL_SIGNALS, json.dumps(signal))
            logger.info(f"Pushed signal: {signal}")

    async def push_control(self, command):
        if self.r:
            await self.r.publish(self.CHANNEL_CONTROL, json.dumps(command))
            logger.info(f"Pushed control command: {command}")

    async def listen_control(self):
        if not self.r:
            return
        pubsub = self.r.pubsub()
        await pubsub.subscribe(self.CHANNEL_CONTROL)
        while True:
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    yield json.loads(message["data"])
                    logger.info(f"Received control message: {message['data']}")
            except Exception as e:
                logger.error(f"Error listening to control channel: {e}")
            await asyncio.sleep(0.1)


streamor = StreamManager()
