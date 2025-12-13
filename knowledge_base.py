import logging
import time
from typing import cast

from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, QueryResponse, ServerlessSpec

from config import settings

logger = logging.getLogger("KnowledgeBase")


class MarketMemory:
    def __init__(self):
        self.pc = None
        self.index = None
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def init_pinecone(self):
        if not settings.PINECONE_API_KEY:
            return
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            if settings.PINECONE_INDEX not in [i.name for i in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=settings.PINECONE_INDEX,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            self.index = self.pc.Index(settings.PINECONE_INDEX)
        except Exception as e:
            logger.error(f"Pinecone Error: {e}")

    def store_memory(self, text: str, metadata: dict):
        if self.index:
            vec = self.embedder.embed_query(text)
            self.index.upsert(
                vectors=[(f"mem_{int(time.time())}", vec, {"text": text, **metadata})]
            )

    def recall_memory(self, query: str):
        if self.index:
            vec = self.embedder.embed_query(query)
            res = cast(
                QueryResponse,
                self.index.query(vector=vec, top_k=2, include_metadata=True),
            )
            return "\n".join([m["metadata"]["text"] for m in res.matches])
        return ""


memory_bank = MarketMemory()
