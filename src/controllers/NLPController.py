from typing import List

from models.db_schemas import DataChunk
from stores.llm.LLMEnums import DocumentType
from helpers.config import get_settings

from .BaseController import BaseController
from models.db_schemas import Project
import json
class NLPController(BaseController):
    
    def __init__(self, vectordb_client,generation_client,embedding_client,template_parser):
        super().__init__()
        self.vectordb_client=vectordb_client
        self.generation_client=generation_client
        self.embedding_client=embedding_client
        self.template_parser=template_parser
        
        
    def create_collection_name(self,project_id:int):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self,project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name)
    
    async def get_vector_db_collection_info(self ,project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        collection_info=await self.vectordb_client.get_collection_info(collection_name)
        
        return json.loads(
            json.dumps(collection_info,default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self,project:Project,
                             chunks:List[DataChunk],
                             chunk_ids:List[int],
                             do_reset:bool=False
                             ):
        collection_name=self.create_collection_name(project_id=project.project_id)
        filtered_chunks=[c for c in chunks if c.chunk_text and len(c.chunk_text.strip()) > 1]
        texts=[c.chunk_text.strip() for c in filtered_chunks]
        metadata=[self._parse_chunk_metadata(c.chunk_metadata) for c in filtered_chunks]
        record_ids=[c.chunk_id for c in filtered_chunks]

        if not texts:
            return True

        vectors=self.embedding_client.embed_text(text=texts,document_type=DocumentType.DOCUMENT.value)

        if not vectors:
            return False

        
        _= await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
            
            
        )
        _= await self.vectordb_client.insert_many(collection_name=collection_name,
                                            texts=texts,
                                            metadata= metadata,
                                            vector=vectors,
                                            record_ids=record_ids,
                                            batch_size=get_settings().VECTOR_INSERT_BATCH_SIZE
                                            )
        
        
        return True
    
    
    async def search_vector_db_collection(self,project:Project,text:str,limit:int = 10,
                                          score_threshold:float=None,
                                          metadata_filter:dict=None):
        
        
        query_vector=None
        collection_name=self.create_collection_name(project_id=project.project_id)
        vectors = self.embedding_client.embed_text(
        text=text,
        document_type=DocumentType.QUERY.value
        )
        
        if not vectors or len(vectors)==0:
            return None
        
        
        if isinstance(vectors,list) and len(vectors)>0:
            query_vector=vectors[0]
            
            
        if not query_vector:
            return None
        
        results= await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter
        )
        
        if not results:
            return []
        
        return json.loads(
            json.dumps(results,default=lambda x: x.__dict__)
        )
        
        
    async def answer_rag_question(self,project:Project,query:str,limit:int =5,
                                  score_threshold:float=None,
                                  metadata_filter:dict=None,
                                  include_sources:bool=True):
        
        answer , full_prompt , chat_history=None,None,None
        settings=get_settings()
        
        collection_name=self.create_collection_name(project_id=project.project_id)
        retrived_document= await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=max(limit, settings.RETRIEVAL_TOP_K),
            score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD,
            metadata_filter=metadata_filter
        )
        
        if not retrived_document or len(retrived_document)==0:
            return "", None, None, []
        
        
        
        system_prompt=self.template_parser.get("rag","system_prompt")
        
        
        selected_documents=self._select_documents_for_prompt(
            retrived_document=retrived_document,
            max_documents=limit or settings.ANSWER_TOP_K,
            max_context_chars=settings.MAX_CONTEXT_CHARS
        )

        documnets_prompts="\n".join([
                self.template_parser.get("rag","document_prompt",{
                    "doc_num":idx+1,
                    "chunk_text":self.generation_client.process_text(doc["text"])
                })
            for idx,doc in enumerate(selected_documents)
        ])
        
        footer_prompt=self.template_parser.get("rag","footer_prompt")
        
        chat_history = [
        self.generation_client.construct_prompt(
            prompt=system_prompt,
            role=self.generation_client.enums.SYSTEM.value,
                        )
                    ]

        full_prompt = "\n\n".join([
            documnets_prompts,
            f"## User Question:\n{query}",
            footer_prompt,
        ])
            
        answer=self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )
        
        sources=self._build_sources(selected_documents) if include_sources else []
        return answer , full_prompt , chat_history, sources

    def _parse_chunk_metadata(self, metadata):
        if not metadata:
            return {}

        if isinstance(metadata, dict):
            return metadata

        try:
            return json.loads(metadata)
        except Exception:
            return {}

    def _select_documents_for_prompt(self, retrived_document:list, max_documents:int, max_context_chars:int):
        selected_documents=[]
        used_context_chars=0
        seen_documents=set()

        for doc in retrived_document:
            metadata=doc.get("metadata") or {}
            dedupe_key=(
                metadata.get("asset_id"),
                metadata.get("chunk_order"),
                doc.get("text")
            )

            if dedupe_key in seen_documents:
                continue

            text=doc.get("text") or ""
            next_context_size=used_context_chars + len(text)

            if selected_documents and next_context_size > max_context_chars:
                continue

            seen_documents.add(dedupe_key)
            selected_documents.append(doc)
            used_context_chars += len(text)

            if len(selected_documents) >= max_documents:
                break

        return selected_documents

    def _build_sources(self, selected_documents:list):
        sources=[]

        for idx, doc in enumerate(selected_documents):
            metadata=doc.get("metadata") or {}
            text=doc.get("text") or ""
            sources.append({
                "doc_num": idx + 1,
                "chunk_id": doc.get("chunk_id"),
                "file_name": metadata.get("file_name") or metadata.get("file_id"),
                "asset_id": metadata.get("asset_id"),
                "chunk_order": metadata.get("chunk_order"),
                "score": doc.get("score"),
                "text": text[:300]
            })

        return sources
        
