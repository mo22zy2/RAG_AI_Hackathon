from typing import Dict, Optional

from pydantic import BaseModel
from sqlalchemy import Index

from .breathx_rag_base import SQLAlchemyBase
from sqlalchemy import ForeignKey, String, Column,Integer,DateTime, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
import uuid


class DataChunk(SQLAlchemyBase):
    __tablename__='chunks'
    
    chunk_id=Column(Integer,primary_key=True,autoincrement=True)
    chunk_uuid=Column(UUID(True),default=uuid.uuid4,unique=True,nullable=False)
        
    chunk_text=Column(String,nullable=False)
    chunk_metadata=Column(String,nullable=True)
    chunk_order=Column(Integer,nullable=False)
    
    
    chunk_project_id=Column(Integer,ForeignKey('projects.project_id'),nullable=False)
    chunk_asset_id = Column(Integer,ForeignKey('assets.asset_id'),nullable=False)
    
    project=relationship("Project",back_populates='chunks')
    asset=relationship('Asset',back_populates='chunks')
    
    created_at=Column(DateTime(timezone=True),server_default=func.now(),nullable=False)
    updated_at=Column(DateTime(timezone=True),default=func.now(),server_default=func.now(),onupdate=func.now(),nullable=False)
    
     
    __table_args__=(
        Index('ix_chunk_project_id',chunk_project_id),
        Index('ix_chunk_asset_id',chunk_asset_id),
    )
    
class RetrivedDocument(BaseModel):
    text: str
    score: float
    chunk_id: Optional[int] = None
    metadata: Optional[Dict] = None
