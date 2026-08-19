from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import field_validator
from typing import List, Optional
from pathlib import Path
import json

class Settings (BaseSettings):
    APP_NAME:str
    APP_VERSION:str
    FILE_ALLOWED_TYPES:list
    FILE_MAX_SIZE:int
    FILE_DEFAULT_CHUNK_SIZE:int
    
    
    MONGODB_URL:Optional[str]=None
    MONGODB_DATABASE:Optional[str]=None
    
    POSTGRES_USERNAME:str
    POSTGRES_PASSWORD:str
    POSTGRES_HOST:str
    POSTGRES_PORT:int
    POSTGRES_MAIN_DATABASE:str
    
    GENERATION_BACKEND:str
    EMBEDDING_BACKEND:str

    OPENAI_API_KEY:Optional[str]=None
    OPENAI_BASE_URL:Optional[str]=None


    COHERE_API_KEY:Optional[str]=None

    GENERATION_MODEL_ID_LIST:Optional[str]=None
    GENERATION_MODEL_ID:Optional[str]=None
    EMBEDDING_MODEL_ID:Optional[str]=None
    EMBEDDING_MODEL_SIZE:Optional[int]=None


    INPUT_DEFAULT_MAX_CHARS:Optional[int]=None
    GENERATION_DEFAULT_MAX_TOKENS:Optional[int]=None
    GENERATION_DEFAULT_TEMPERATURE:float=0.8

    RETRIEVAL_TOP_K:int=8
    ANSWER_TOP_K:int=8
    RETRIEVAL_SCORE_THRESHOLD:float=0.0
    ANSWER_MIN_TOP_SCORE:float=0.0
    ANSWER_MIN_EVIDENCE_COUNT:int=1
    ANSWER_MAX_VERIFICATION_REGENERATIONS:int=1
    MAX_CONTEXT_CHARS:int=12000
    EMBEDDING_BATCH_SIZE:int=64
    VECTOR_INSERT_BATCH_SIZE:int=64

    # Retrieval optimization
    RERANK_BACKEND:str="COHERE"
    RERANK_MODEL_ID:str="rerank-v3.5"
    RERANK_TOP_K:int=5
    HYBRID_FUSION:str="rrf"
    HYBRID_RRF_K:int=60

    # Clinical ingestion / section-aware chunking
    DEFAULT_CHUNKING_METHOD:str="section"
    CHUNK_MIN_TOKENS:Optional[int]=400
    CHUNK_MAX_TOKENS:Optional[int]=800
    CHUNK_CHARS_PER_TOKEN:float=4.0
    USE_TIKTOKEN_CHUNKING:bool=False
    TIKTOKEN_ENCODING:str="cl100k_base"
    RUNNING_HEADER_MIN_PAGE_FRACTION:float=0.5
    STRIP_PATTERNS:Optional[List[str]]=None
    SECTION_HEADER_PATTERNS:Optional[List[str]]=None

    @field_validator("STRIP_PATTERNS", "SECTION_HEADER_PATTERNS", mode="before")
    @classmethod
    def _parse_pattern_list(cls, value):
        # pydantic-settings tries JSON first; regex escapes like \s break that,
        # so fall back to a comma-separated list.
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [
                item.strip().strip('"').strip("'")
                for item in value.split(",")
                if item.strip()
            ]
        return value


    VECTOR_DB_BACKEND_LITERAL:Optional[List[str]]=None
    VECTOR_DB_BACKEND:str
    VECTOR_DB_PATH:str
    VECTOR_DB_DISTANCE_METHOD:str
    VECTOR_DB_PGVEC_INDEX_THRESHOLD:int
    VECTOR_DB_INDEX_TYPE:str
    
    DEFAULT_LANGUAGE:str
    
    class Config:
        env_file=Path(__file__).parent.parent / '.env'
        
def get_settings():
    return Settings()
