import json
import logging
import redis.asyncio as redis
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StateManager")


class StateManager:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.KEY_STATE = f"bot_state:{settings.ACTIVE_SYMBOL}"

    async def get_active_position(self):
        try:
            data = await self.r.get(self.KEY_STATE)
            return json.loads(data) if data else None
        except:
            return None

    async def set_active_position(self, symbol, side, price, lot, tp, sl):
        state = {
            "symbol": symbol,
            "side": side,
            "entry_price": float(price),
            "lot": float(lot),
            "tp": float(tp),
            "sl": float(sl),
            "status": "OPEN",
        }
        await self.r.set(self.KEY_STATE, json.dumps(state))

    async def clear_active_position(self):
        await self.r.delete(self.KEY_STATE)


state_manager = StateManager()
