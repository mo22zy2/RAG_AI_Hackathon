from typing import List

from ..RerankProviderInterface import RerankProviderInterface
from models.db_schemas import RetrivedDocument
import cohere
import logging


class CoHereRerankProvider(RerankProviderInterface):

    def __init__(self, api_key: str, model_id: str = "rerank-v3.0"):
        self.api_key = api_key
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        # AsyncClient so a rerank call doesn't block the event loop.
        self.client = cohere.AsyncClient(self.api_key, timeout=240) if api_key else None

    async def rerank(self, query: str, documents: List[RetrivedDocument], top_n: int = None) -> List[RetrivedDocument]:
        if not self.client:
            self.logger.error("Cohere rerank client is not initialized.")
            return documents

        if not documents:
            return []

        texts = [doc.text for doc in documents]
        top_n = top_n or len(texts)

        try:
            response = await self.client.rerank(
                model=self.model_id,
                query=query,
                documents=texts,
                top_n=min(top_n, len(texts)),
            )
        except Exception as e:
            self.logger.error(f"Cohere rerank failed, returning original order: {e}")
            return documents

        if not response or not response.results:
            self.logger.error("No results returned from Cohere rerank.")
            return documents

        reranked = []
        for result in response.results:
            doc = documents[result.index]
            reranked.append(
                RetrivedDocument(
                    text=doc.text,
                    score=result.relevance_score,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                )
            )

        return reranked
