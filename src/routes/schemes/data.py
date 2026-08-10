from pydantic import BaseModel
from typing import Optional

class ProccessRequest(BaseModel):
    file_id:str=None
    chunck_size:Optional[int]=100
    overlap_size:Optional[int]=20
    do_reset:Optional[int]=0
    chunking_method: Optional[str] = "recursive"
    
