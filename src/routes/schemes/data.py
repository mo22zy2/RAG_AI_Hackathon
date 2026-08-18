from pydantic import BaseModel
from typing import Optional

class ProccessRequest(BaseModel):
    file_id:str=None
    chunk_size:Optional[int]=100
    overlap_size:Optional[int]=50
    do_reset:Optional[int]=0
    chunking_method: Optional[str] = None
    # Provenance fallback, applied when the asset has no asset_config
    document_name: Optional[str] = None
    source_url: Optional[str] = None
    org: Optional[str] = None
    
