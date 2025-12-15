import asyncio
import json

import redis.asyncio as redis
from portfolio_manager import portfolio

from config import settings
from logging_config import setup_logger

logger = setup_logger("FusionEngine")


class FusionEngine:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

    async def process(self, msg):
        dna = json.loads(await self.r.get("system:dna") or "{}")
        w_tech = dna.get("w_tech", 0.7)
        w_rl = dna.get("w_rl", 1.0)

        signal = json.loads(msg["data"])
        source = signal.get("source")
        conf = signal.get("confidence", 0.5)
        action = signal.get("action")

        # Scoring Logic
        score = 0
        if action == "BUY":
            score += conf * (w_rl if source == "BRAIN_RL" else w_tech)
        elif action == "SELL":
            score -= conf * (w_rl if source == "BRAIN_RL" else w_tech)

        # Social & LLM Logic (Simplified)
        llm = json.loads(await self.r.get("llm:analysis") or "{}")
        if llm.get("action") == "BUY":
            score += 0.3

        final_act = "HOLD"
        if score > 0.6:
            final_action = "BUY"
        elif score < -0.6:
            final_action = "SELL"
        else:
            return

        # Portfolio Check
        allowed, units, reason = await portfolio.calculate_allocation(
            signal["symbol"], conf
        )
        if allowed:
            payload = {
                "action": final_action,
                "symbol": signal["symbol"],
                "units": units,
                "reason": reason,
            }
            await self.r.publish(settings.CHANNEL_AI_ANALYSIS, json.dumps(payload))

    async def run(self):
        ps = self.r.pubsub()
        await ps.subscribe(settings.CHANNEL_SIGNALS)
        logger.info("⚖️ Fusion Engine Active")
        async for msg in ps.listen():
            if msg["type"] == "message":
                await self.process(msg)


if __name__ == "__main__":
    asyncio.run(FusionEngine().run())
