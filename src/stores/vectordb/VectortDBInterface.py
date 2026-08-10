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
    
    
    
