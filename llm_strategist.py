import asyncio
import json

import redis.asyncio as redis
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import SecretStr

from config import settings
from logging_config import setup_logger

logger = setup_logger("LLM-Strategist")


class LLMStrategist:
    def __init__(self):
        self.r = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )
        self.llm_primary = HuggingFaceEndpoint(
            model=settings.HF_MODEL_ID,
            huggingfacehub_api_token=settings.HUGGINGFACE_API_TOKEN,
            temperature=0.1,
        )
        self.llm_backup = ChatGroq(
            api_key=SecretStr(settings.GROQ_API_KEY),
            model=settings.GROQ_MODEL_ID,
            temperature=0.1,
        )

    async def analyze(self):
        sentiment = await self.r.get("latest_sentiment") or "Neutral"
        macro = await self.r.get("macro:status") or "Safe"
        prompt_template = PromptTemplate.from_template(
            "Analyze market. Symbol: {symbol}, Sentiment: {sentiment}, Macro: {macro}. JSON Output: {{action, reasoning, confidence}}"
        )
        prompt = prompt_template.format(
            symbol=settings.ACTIVE_SYMBOLS[0], sentiment=sentiment, macro=macro
        )

        try:
            logger.info("🧠 LLM Thinking...")
            res = self.llm_primary.invoke(prompt)
        except Exception:
            logger.warning("⚠️ Primary Fail, Switching to Groq...")
            res = self.llm_backup.invoke(prompt).content

        # Simple JSON extraction logic needed here in production
        import re

        match = re.search(r"\{.*\}", str(res), re.DOTALL)
        if match:
            await self.r.set("llm:analysis", match.group(0))

    async def run(self):
        while True:
            await self.analyze()
            await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(LLMStrategist().run())
