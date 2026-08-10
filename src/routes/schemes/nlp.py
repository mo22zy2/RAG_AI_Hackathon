from pydantic import BaseModel
from typing import Optional


class PushRequest(BaseModel):
    do_reset: Optional[int]=0
    
    
class SearchRequest(BaseModel):
    text:str
    limit:Optional[int]=5
    score_threshold: Optional[float] = None
    metadata_filter: Optional[dict] = None
    include_sources: Optional[bool] = True
    retrieval_mode: Optional[str] = "vector"
