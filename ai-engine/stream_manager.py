import json
import logging

import redis
from config import settings

logger = logging.getLogger("StreamManager")


class StreamManager:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.STREAM_MARKET = "stream:market_data"
        self.STREAM_SIGNALS = "stream:trade_signals"
        self.GROUP_AI = "group_ai_engine"

    def push_market_data(self, symbol, data_dict):
        try:
            self.r.xadd(
                self.STREAM_MARKET, {"symbol": symbol, "data": json.dumps(data_dict)}
            )
            self.r.xtrim(self.STREAM_MARKET, maxlen=1000)
        except Exception as e:
            logger.error(f"Redis Push Error: {e}")

    def consume_market_data(self, consumer_name):
        try:
            self.r.xgroup_create(
                self.STREAM_MARKET, self.GROUP_AI, id="0", mkstream=True
            )
        except:
            pass
        entries = self.r.xreadgroup(
            self.GROUP_AI, consumer_name, {self.STREAM_MARKET: ">"}, count=1, block=1000
        )
        parsed = []
        for _, messages in entries:
            for msg_id, content in messages:
                self.r.xack(self.STREAM_MARKET, self.GROUP_AI, msg_id)
                parsed.append(json.loads(content["data"]))
        return parsed

    def push_signal(self, signal_dict):
        self.r.publish("channel_signals", json.dumps(signal_dict))
        self.r.xadd(self.STREAM_SIGNALS, signal_dict)


streamor = StreamManager()
