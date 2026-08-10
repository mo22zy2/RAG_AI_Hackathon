from sqlalchemy import func, select ,delete

from .BaseDataModel import BaseDataModel
from .db_schemas import DataChunk

class ChunkModel(BaseDataModel):
    
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client=db_client

    # Now i was facing a problem that the init in Python should not be async, so I created a class method to create an instance of the class and initialize the collection asynchronously.
    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance
        
    async def create_chunk(self, chunk: DataChunk):
        async with self.db_client() as session:
            async with session.begin():
                session.add(chunk)
            await session.refresh(chunk)
        return chunk
    
    
    async def get_chunk_by_id(self, chunk_id: str):
        async with self.db_client() as session:
            async with session.begin():
                
                result = await session.execute(select(DataChunk).where(DataChunk.chunk_id==chunk_id))
                chunk=result.scalar_one_or_none()
                
            return chunk
    
    async def insert_many_chunks(self,chunks:list, batch_size:int=100):
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0,len(chunks),batch_size):
                    batch = chunks[i:i+batch_size]
                    session.add_all(batch)
                                    
        return len(chunks)
    
    async def delete_chunk_by_project_id(self,project_id:int):
        async with self.db_client() as session:
            delete_query=delete(DataChunk).where(DataChunk.chunk_project_id==project_id)
            result=await session.execute(delete_query)
            await session.commit()
        return result.rowcount
    
    async def get_project_chunks(self, project_id:int,page_no: int=1,page_size:int=50):
       
        async with self.db_client()as session:
            get_project_chunk_query=select(DataChunk).where(DataChunk.chunk_project_id==project_id).offset((page_no-1)*page_size).limit(page_size)
            result=await session.execute(get_project_chunk_query)
            record=result.scalars().all()
            
        return record
    
    
    async def get_total_chunk(self, project_id: int):
        async with self.db_client() as session:
            async with session.begin():
                count_sql = select(func.count()).where(DataChunk.chunk_project_id == project_id)
                records_count = await session.execute(count_sql)
                total_count = records_count.scalar()
                
        return total_count