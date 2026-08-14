from abc import abstractmethod,ABC
from typing import List
from models.db_schemas import RetrivedDocument


class VectorDBInterface(ABC):
    
    
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def is_collection_existed(self, collection_name: str) -> bool:
        pass
    
    def list_all_collections(self) -> List:
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> dict:
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int, do_reset:bool=False):
        pass
    
    @abstractmethod
    def insert_one(self, collection_name: str,
                   vector:list,
                   text:str,
                   metadata: dict=None,
                   record_id: str=None):
        pass
    
    @abstractmethod
    def insert_many(self, collection_name: str,
                   vector:list,
                    texts:str,
                   metadata: dict=None,
                   record_ids: str=None,
                   batch_size:int=50):
        pass
    
    @abstractmethod
    def search_by_vector(self, collection_name: str,
                         vector:list,
                         limit:int,
                         score_threshold:float=None,
                         metadata_filter:dict=None)->List[RetrivedDocument]:
        pass

    def search_by_keyword(self, collection_name: str,
                          query: str,
                          limit: int,
                          metadata_filter: dict = None,
                          ts_config: str = "english") -> List[RetrivedDocument]:
        """
        Keyword (BM25-style ts_rank) search. Implemented by providers that
        support full-text search (e.g. PGVector). Providers without support
        raise NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support keyword search")

    def search_hybrid(self, collection_name: str,
                      vector: list,
                      query: str,
                      limit: int,
                      metadata_filter: dict = None,
                      rrf_k: int = 60,
                      ts_config: str = "english") -> List[RetrivedDocument]:
        """
        Hybrid search combining vector similarity and keyword search
        (e.g. Reciprocal Rank Fusion). Providers without support raise
        NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support hybrid search")
    
    
    
