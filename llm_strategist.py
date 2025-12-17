import asyncio
import json
import logging

import redis.asyncio as redis
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

from config import settings
from knowledge_base import memory_bank
from logging_config import setup_logger

logger = setup_logger("LLM-Strategist")


class LLMStrategist:
    def __init__(self):
        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.r = redis.Redis(**kwargs)
        # Menggunakan Llama 3 70B (Versi paling powerful di Groq saat ini)
        self.llm = ChatGroq(
            api_key=SecretStr(settings.GROQ_API_KEY),
            model=settings.GROQ_MODEL_ID,
            temperature=0.3,  # Rendah agar lebih analitis, bukan kreatif
        )

    async def analyze_market_context(self):
        # 1. Ambil Data Terkini
        sentiment_raw = (
            await self.r.get("sentiment_latest") or "{}"
        )  # Dari Sentiment Engine
        macro_status = await self.r.get("macro:status") or "Neutral"

        try:
            sent_data = json.loads(sentiment_raw)
            news_summary = sent_data.get("summary", "No significant news.")
            sent_score = sent_data.get("sentiment_score", 0)
        except:
            news_summary = "No news data"
            sent_score = 0

        # 2. RAG (Retrieval-Augmented Generation) - Cek Memori
        # Cari memori tentang kondisi serupa
        query = f"Market condition with sentiment score {sent_score} and news: {news_summary}"
        past_lessons = memory_bank.recall_similar_situations(query)

        # 3. Construct Prompt Enterprise
        template = """
        You are a Senior Hedge Fund Risk Manager. Analyze the current market for {symbol}.
        
        === CURRENT DATA ===
        News Summary: {news}
        Sentiment Score: {score} (-1.0 Bearish to 1.0 Bullish)
        Macro Status: {macro}
        
        === PAST LESSONS (MEMORY) ===
        {memory}
        
        === TASK ===
        Decide the STRATEGIC BIAS (BULLISH/BEARISH/NEUTRAL) and CONFIDENCE (0.0 - 1.0).
        Consider past lessons carefully. If similar situations led to losses, suggest CAUTION.
        
        OUTPUT JSON ONLY: {{"bias": "string", "confidence": float, "reasoning": "string"}}
        """

        prompt = PromptTemplate.from_template(template).format(
            symbol=settings.ACTIVE_SYMBOLS[0],
            news=news_summary,
            score=sent_score,
            macro=macro_status,
            memory=past_lessons,
        )

        try:
            logger.info("🧠 LLM Strategist Thinking (with RAG)...")
            response = await self.llm.ainvoke(prompt)
            content = response.content

            # Parsing JSON (Robust)
            import re

            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            if match:
                analysis = json.loads(match.group(0))
                # Simpan hasil analisis ke Redis untuk dibaca Fusion Engine
                await self.r.set("llm:analysis", json.dumps(analysis))
                logger.info(
                    f"💡 LLM Verdict: {analysis['bias']} ({analysis['confidence']:.2f})"
                )

        except Exception as e:
            logger.error(f"LLM Strategy Error: {e}")

    async def run(self):
        memory_bank.init_pinecone()
        logger.info("🤖 LLM Strategist Active (RAG Enabled)")
        while True:
            await self.analyze_market_context()
            # Analisis ulang setiap 15 menit (Fundamental tidak berubah detik-an)
            await asyncio.sleep(900)


if __name__ == "__main__":
    asyncio.run(LLMStrategist().run())
