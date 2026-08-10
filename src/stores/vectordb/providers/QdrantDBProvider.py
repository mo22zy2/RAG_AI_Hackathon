from typing import List

from ..VectortDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnum
from qdrant_client import models, QdrantClient # type: ignore
from models.db_schemas import RetrivedDocument
import logging



class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_client:str, default_vector_size: int = 786, distance_method: str = None,index_threshold:int=100):
        self.client=None
        self.db_client=db_client
        self.distance_method=None
        self.default_vector_size=default_vector_size
        
        if distance_method==DistanceMethodEnum.COSINE.value:
            self.distance_method=models.Distance.COSINE
        elif distance_method==DistanceMethodEnum.DOT.value:
            self.distance_method=models.Distance.DOT
        elif distance_method==DistanceMethodEnum.EUCLIDEAN.value:
            self.distance_method=models.Distance.EUCLIDEAN
            
            
        self.logger=logging.getLogger('uvicorn')
        
        
    async def connect(self):
        self.client=QdrantClient(path=self.db_client)
        self.logger.info(f"Connected to QdrantDB at {self.db_client}")
        
    async def disconnect(self):
        self.client=None
        
    async def is_collection_existed(self, collection_name:str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    async def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    async def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)
    
    async def delete_collection(self, collection_name: str):
        if await self.is_collection_existed(collection_name=collection_name):
            self.logger.info(f"Deleting collection: {collection_name}")
            
            return self.client.delete_collection(collection_name=collection_name)
    
    async def create_collection(self, collection_name: str, embedding_size: int, do_reset:bool=False):
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)
            
        if not await self.is_collection_existed(collection_name=collection_name):
          
            self.logger.info(f'Creating new Qdrant collection: {collection_name}')
            
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_size, distance=self.distance_method)
            )
            return True
        return False
    
    async def insert_one(self, collection_name: str,
                   vector:list,
                   text:str,
                   metadata: dict=None,
                   record_id: str=None):
        if not await self.is_collection_existed(collection_name=collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        
        try:
            _=self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=record_id,
                        vector=vector,
                        payload={
                                "text":text,
                                "metadata":metadata}
                    )
                ]
            )
            
        except Exception as e:
            self.logger.error(f"Error occures while inserting batch : {e}")
            return False
            
        return True
    
    
    async def insert_many(self, collection_name: str,
                   vector:list,
                   texts:str,
                   metadata: dict=None,
                   record_ids: str=None,
                   batch_size:int=50):
        
        if not await self.is_collection_existed(collection_name=collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        
        if metadata is None:
            metadata=[None]*len(texts)
            
        if record_ids is None:
            record_ids=list(range(0,len(texts)))
        
        
        for i in range(0,len(texts),batch_size):
            batch_end=i+batch_size

            batch_texts=texts[i:batch_end]
            batch_vector=vector[i:batch_end]
            batch_metadata=metadata[i:batch_end]
            batch_record_ids=record_ids[i:batch_end]
            
            batch_records=[
                
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vector[x],
                    payload={
                             "text":batch_texts[x],
                             "metadata":batch_metadata[x]}
                )
                
                for x in range(len(batch_texts))
            ]     
            
            
            try:       
                _=self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                    )
            except Exception as e:
                self.logger.error(f"Error occures while inserting batch : {e}")
                return False

        return True
    
    
    async def search_by_vector(self, collection_name: str,
                        vector:list,
                        limit:int=5,
                        score_threshold:float=None,
                        metadata_filter:dict=None):
        query_filter = None
        if metadata_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                    for key, value in metadata_filter.items()
                ]
            )
        
        results= self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        )
        
        
        if not results or len(results)==0:
            return None

        
        return [
            RetrivedDocument(**{
                                "score":result.score,
                                "text":result.payload["text"],
                                "chunk_id":result.id,
                                "metadata":result.payload.get("metadata") or {}
                                })
            for result in results
        ]
