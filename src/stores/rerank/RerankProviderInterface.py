from abc import ABC, abstractmethod
from typing import List

from models.db_schemas import RetrivedDocument


class RerankProviderInterface(ABC):

    @abstractmethod
    async def rerank(self, query: str, documents: List[RetrivedDocument], top_n: int = None) -> List[RetrivedDocument]:
        """
        Reorder `documents` by relevance to `query`. Returned documents keep
        their original order fields but carry updated `.score` values
        (provider relevance scores). Returns at most `top_n` documents.
        """
        pass
