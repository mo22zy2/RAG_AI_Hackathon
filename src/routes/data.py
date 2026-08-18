from fastapi import APIRouter,FastAPI,Depends,UploadFile,status,Request,Form
from fastapi.responses import JSONResponse
from helpers.config import get_settings,Settings
from controllers import DataController,ProjectController,ProcessController
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from models import Response
from .schemes.data import ProccessRequest
from models.db_schemas import DataChunk,Asset
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers import NLPController

import os
import aiofiles
import json
import logging
from typing import Optional

logger=logging.getLogger("uvicorn.error")

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1,data"]
)

@data_router.post("/upload/{project_id}")

async def upload_data(
    request:Request,
    project_id:int,
    file:UploadFile,
    document_name: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    org: Optional[str] = Form(None),
    app_settings:Settings =Depends(get_settings)):
    
    project_model=await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    
    project= await project_model.get_project_or_create_one(project_id=project_id)
    
    data_controller=DataController()
    isValid,response_signal = await data_controller.validate_upload_file(file=file)
    
    if not isValid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":response_signal}
        )
        
    project_dir_path=ProjectController().get_project_path(project_id=project_id)
    
    os.makedirs(project_dir_path, exist_ok=True)
    
    file_path,file_id=data_controller.generate_unique_filepath(original_file_name=file.filename,
                                                       project_id=project_id,
                                                       )
    max_size_bytes = app_settings.FILE_MAX_SIZE * 1048576
    total_bytes = 0
    try:
        
        async with aiofiles.open(file_path,"wb")as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    await f.close()
                    os.remove(file_path)
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"signal":Response.FILE_SIZE_EXCEEDED.value}
                    )
                await f.write(chunk)
                
    except Exception as e:
        logger.error(f"Error while uploading file : {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":Response.FILE_VALIDATED_FALIED.value}
        )
        
        
    asset_model=await AssetModel.create_instance(db_client=request.app.db_client)

    asset_config = {}
    if document_name:
        asset_config["document_name"] = document_name
    if source_url:
        asset_config["source_url"] = source_url
    if org:
        asset_config["org"] = org

    asset_resource=Asset(
        asset_project_id=project.project_id,
        asset_name=file_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_size=str(os.path.getsize(file_path)),
        asset_config=asset_config or None,
    )
    
    asset_record= await asset_model.create_asset(asset=asset_resource)
    
        
    return JSONResponse(
        content={
            "signal":Response.FILE_UPLOAD_SUCCED.value,
            "file_id":str(asset_record.asset_id),

        }
    )
    
    
    
@data_router.post("/process/{project_id}")

async def process_endpoint(request:Request,project_id:int,process_request:ProccessRequest):
    
    chunk_size=process_request.chunk_size
    overlap_size=process_request.overlap_size
    do_reset=process_request.do_reset
    app_settings=get_settings()
    
    project_model=await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    
    project= await project_model.get_project_or_create_one(project_id=project_id)
    
    nlp_controller=NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client
    )
    
    
    asset_model=await AssetModel.create_instance(db_client=request.app.db_client)

    project_file_ids={}
    asset_configs={}
    
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.project_id,
            asset_name=process_request.file_id
        )
        
        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "signal":Response.FILE_ID_ERROR.value,  
        }
                            )
        project_file_ids={
            asset_record.asset_id: asset_record.asset_name
            
        }
        asset_configs={
            asset_record.asset_id: asset_record.asset_config or {}
        }
        
            
        # project_file_ids=[process_request.file_id]
    else:
        asset_model=await AssetModel.create_instance(db_client=request.app.db_client)

        project_files = await asset_model.get_all_project_assets(
    asset_project_id=project.project_id,
    asset_type=AssetTypeEnum.FILE.value
)
        project_file_ids = {
    record.asset_id: record.asset_name
    for record in project_files
}
        asset_configs = {
    record.asset_id: record.asset_config or {}
    for record in project_files
}
        
    if len(project_file_ids)==0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":Response.NO_FILES_ERROR.value}
    )
    
    
    process_controller=ProcessController(project_id)
    chunk_model=await ChunkModel.create_instance(db_client=request.app.db_client)
    
#First retriving the Collection name then call delete collection
    if do_reset==1:
        collection_name=nlp_controller.create_collection_name(project_id=project.project_id)
       
        _= await request.app.vectordb_client.delete_collection(collection_name=collection_name)
       
        _= await chunk_model.delete_chunk_by_project_id(project_id=project.project_id)
   
   
    no_records=0
    no_files=0
    
    
    for asset_id , file_id in project_file_ids.items():
        
    
        file_content=process_controller.get_file_content(file_id=file_id)
        
        if file_content is None:
            logger.error(f"Error while Processing File {file_id}")
            continue
        
        file_chunks=process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            method=process_request.chunking_method or app_settings.DEFAULT_CHUNKING_METHOD
            )
        
        if file_chunks is None or len(file_chunks)==0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal":Response.FILE_PROCESSING_FALIED.value}
                )
            
        asset_config = asset_configs.get(asset_id, {})
        provenance = {
            "document_name": asset_config.get("document_name")
            or process_request.document_name
            or file_id,
            "source_url": asset_config.get("source_url")
            or process_request.source_url
            or "",
            "org": asset_config.get("org") or process_request.org or "",
        }

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=json.dumps({
                    **chunk.metadata,
                    "asset_id": asset_id,
                    "file_name": file_id,
                    "project_id": project.project_id,
                    "chunk_order": i+1,
                    **provenance,
                }),
                chunk_order=i+1,
                chunk_project_id=project.project_id,
                chunk_asset_id=asset_id
        )
            for i,chunk in enumerate(file_chunks)]
        
        
        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files+=1
    return JSONResponse(
        content={
            "signal":Response.FILE_PROCESSING_SUCCEED.value,
            "inserted_chunks":no_records,
            "processed_files":no_files
            
        }
    )
    
