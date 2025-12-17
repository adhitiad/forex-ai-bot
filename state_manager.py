import datetime
import json
import logging

import redis.asyncio as redis

from config import settings

logger = logging.getLogger("StateManager")


class StateManager:
    def __init__(self):
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r = redis.Redis(**kwargs)
        self.KEY_STATE = f"bot_state:{settings.ACTIVE_SYMBOLS[0]}"
        self.KEY_DAILY_PNL = f"bot_stats:daily_pnl:{datetime.date.today()}"

    async def get_active_position(self):
        d = await self.r.get(self.KEY_STATE)
        return json.loads(d) if d else None

    async def set_active_position(self, symbol, side, price, lot, tp, sl):
        await self.r.set(
            self.KEY_STATE,
            json.dumps(
                {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": float(price),
                    "lot": float(lot),
                    "tp": float(tp),
                    "sl": float(sl),
                    "status": "OPEN",
                }
            ),
        )

    async def clear_active_position(self):
        await self.r.delete(self.KEY_STATE)

    async def update_pnl(self, amount: float):
        await self.r.incrbyfloat(self.KEY_DAILY_PNL, amount)

    async def check_circuit_breaker(self, balance=10000.0):
        pnl = float(await self.r.get(self.KEY_DAILY_PNL) or 0)
        dd = (pnl / balance) * 100
        if dd <= -settings.MAX_DAILY_LOSS_PERCENT:
            return {"status": "STOP", "reason": f"DAILY LOSS LIMIT ({dd:.2f}%)"}
        return {"status": "OK", "reason": ""}

    async def get_trading_mode(self):
        # Check if manual mode is set in Redis
        mode = await self.r.get("trading_mode")
        if mode:
            return mode
        # Default: LIVE if MT5 credentials are set, else PAPER
        return "LIVE" if settings.MT5_LOGIN else "PAPER"

    async def set_trading_mode(self, mode: str):
        # Store the trading mode in Redis (optional, for manual override)
        await self.r.set("trading_mode", mode)

    async def set_asset_config(self, symbol: str, timeframe: str):
        """Simpan config aset baru ke Redis"""
        await self.r.set("config:active_symbol", symbol)
        await self.r.set("config:timeframe", timeframe)

    async def get_asset_config(self):
        """Ambil config aset saat ini (Ingestor/Brain pakai ini)"""
        sym = await self.r.get("config:active_symbol")
        tf = await self.r.get("config:timeframe")

        # Fallback ke settings bawaan jika Redis kosong
        return {
            "symbol": sym if sym else settings.ACTIVE_SYMBOLS[0],
            "timeframe": tf if tf else settings.TIMEFRAME,
        }


state_manager = StateManager()
