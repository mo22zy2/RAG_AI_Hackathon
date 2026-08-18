from pydantic import BaseModel, Field
from typing import Optional, List


class PushRequest(BaseModel):
    do_reset: Optional[int]=0


class ConversationTurn(BaseModel):
    question: str
    answer: Optional[str] = None


class SearchRequest(BaseModel):
    text:str
    limit:Optional[int]=5
    # Prior turns in this session, oldest first, so follow-up questions
    # ("what about someone older?") can be understood in context.
    conversation_history: Optional[List[ConversationTurn]] = None
    score_threshold: Optional[float] = None
    # Swagger UI auto-fills a bare `Optional[dict]` with a placeholder like
    # {"additionalProp1": {}} — an explicit null example stops that, since
    # submitting the placeholder as-is filters on a metadata key that no
    # chunk has and silently returns zero results.
    metadata_filter: Optional[dict] = Field(default=None, examples=[None])
    include_sources: Optional[bool] = True
    # "vector" | "keyword" | "hybrid"
    retrieval_mode: Optional[str] = "hybrid"
    # Post-retrieval rerank. None = auto (enabled for /answer, disabled for /search)
    rerank: Optional[bool] = None
    # Expand asthma abbreviations/synonyms before retrieval.
    expand_query: Optional[bool] = None
    # Run deterministic citation and claim-support checks on generated answers.
    verify_claims: Optional[bool] = True
