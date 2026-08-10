from fastapi import APIRouter, FastAPI, UploadFile, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers.NLPController import NLPController
from tqdm.auto import tqdm

from models import Response
import logging

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix='/api/v1/nlp',
    tags=["api_v1", 'nlp']
)


@nlp_router.post('/index/push/{project_id}')
async def index_project(request: Request, project_id: int, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": Response.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    # Create collection if it does not exist
    collection_name = nlp_controller.create_collection_name(project_id=project_id)

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    # Batching
    total_chunks_count = await chunk_model.get_total_chunk(project_id=project.project_id)

    pbar = tqdm(
        total=total_chunks_count,
        desc="Vector Indexing",
        position=0
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0

    try:
        while has_records:

            page_chunk = await chunk_model.get_project_chunks(
                project_id=project.project_id,
                page_no=page_no
            )

            if not page_chunk:
                has_records = False
                break

            page_no += 1

            chunk_ids = [c.chunk_id for c in page_chunk]

            is_inserted = await nlp_controller.index_into_vector_db(
                project=project,
                chunks=page_chunk,
                chunk_ids=chunk_ids
            )

            if not is_inserted:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "signal": Response.INSERT_INTO_VECTOR_DB_ERROR.value
                    }
                )

            pbar.update(len(page_chunk))
            inserted_items_count += len(page_chunk)
    finally:
        pbar.close()

    return JSONResponse(
        content={
            "signal": Response.INSERT_INTO_VECTOR_DB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )


@nlp_router.get('/index/info/{project_id}')
async def get_project_index_info(request: Request, project_id: int):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": Response.VECTORDB_COLLECTION_RETREIVED.value,
            "collection_info": collection_info
        }
    )


@nlp_router.post('/index/search/{project_id}')
async def search_index_info(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit,
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter
    )

    # Use `is None` rather than falsy check so a legitimate empty
    # result set (e.g. []) isn't reported as an error.
    if results is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": Response.SEARCH_IN_VECTOR_DB_ERROR.value
            }
        )

    return JSONResponse(
        content={
            "signal": Response.SEARCH_IN_VECTOR_DB_SUCCESS.value,
            "results": results
        }
    )


@nlp_router.post('/index/answer/{project_id}')
async def answer_index_info(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    answer, full_prompt, chat_history, sources = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter,
        include_sources=search_request.include_sources
    )

    # `None` means the generation call itself failed.
    if answer is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": Response.RAG_ANSWER_ERROR.value
            }
        )

    # Empty answer means retrieval returned no relevant documents.
    if not answer:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": Response.RAG_NO_DOCUMENTS_FOUND.value,
                'answer': '',
                'sources': sources or [],
            }
        )

    return JSONResponse(
        content={
            "signal": Response.RAG_ANSWER_SUCCEED.value,
            'answer': answer,
            'sources': sources,
            'full_prompt': full_prompt,
            'chat_history': chat_history
        }
    )
