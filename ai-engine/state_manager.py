import json

import redis
from config import settings


class StateManager:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.KEY_POS = "bot:active_position"

    def get_active_position(self):
        data = self.r.get(self.KEY_POS)
        return json.loads(data) if data else None

    def set_active_position(self, symbol, side, entry_price, lot, tp, sl):
        pos_data = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "lot": lot,
            "tp": tp,
            "sl": sl,
        }
        self.r.set(self.KEY_POS, json.dumps(pos_data))

    def clear_active_position(self):
        self.r.delete(self.KEY_POS)


state_manager = StateManager()
