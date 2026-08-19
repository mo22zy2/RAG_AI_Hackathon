from fastapi import APIRouter, FastAPI, UploadFile, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client
    )

    collection_name = nlp_controller.create_collection_name(project_id=project_id)

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    if push_request.do_reset:
        await chunk_model.reset_chunk_indexed(project_id=project.project_id)

    unindexed_count = await chunk_model.get_unindexed_chunk_count(project_id=project.project_id)

    if unindexed_count == 0:
        return JSONResponse(
            content={
                "signal": Response.INSERT_INTO_VECTOR_DB_SUCCESS.value,
                "inserted_items_count": 0,
                "message": "All chunks already indexed"
            }
        )

    pbar = tqdm(
        total=unindexed_count,
        desc="Vector Indexing",
        position=0
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0

    try:
        while has_records:

            page_chunk = await chunk_model.get_unindexed_chunks(
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
                        "signal": Response.INSERT_INTO_VECTOR_DB_ERROR.value,
                        "inserted_items_count": inserted_items_count,
                        "remaining_items": unindexed_count - inserted_items_count,
                    }
                )

            await chunk_model.mark_chunks_indexed(chunk_ids)

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
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client
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
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit,
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter,
        retrieval_mode=search_request.retrieval_mode,
        rerank=bool(search_request.rerank),
        expand_query=search_request.expand_query is True,
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
            "query": search_request.text,
            "expanded_query": nlp_controller.expand_query(search_request.text) if search_request.expand_query is True else search_request.text,
            "query_expanded": search_request.expand_query is True,
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
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client
    )

    use_rerank = search_request.rerank if search_request.rerank is not None else True
    use_query_expansion = search_request.expand_query if search_request.expand_query is not None else True

    conversation_history = [turn.model_dump() for turn in (search_request.conversation_history or [])]

    answer, full_prompt, chat_history, sources, risk_assessment, confidence, quality, disclaimer, evidence_panel, expanded_query = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter,
        include_sources=search_request.include_sources,
        retrieval_mode=search_request.retrieval_mode,
        rerank=use_rerank,
        expand_query=use_query_expansion,
        verify_claims=bool(search_request.verify_claims),
        conversation_history=conversation_history,
    )

    pipeline_metadata = {
        'retrieval_mode': search_request.retrieval_mode,
        'rerank_enabled': use_rerank,
        'query_expansion_enabled': use_query_expansion,
        'verify_claims_enabled': bool(search_request.verify_claims),
    }

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
                'risk_assessment': risk_assessment,
                'confidence': confidence,
                'confidence_summary': NLPController._build_confidence_summary(confidence),
                'evidence_panel': evidence_panel,
                'pipeline_metadata': pipeline_metadata,
                'quality': quality,
                'citations': quality.get("citations", []),
                'unsupported_claims': quality.get("unsupported_claims", []),
                'unsupported_claim_rate': quality.get("unsupported_claim_rate", 0.0),
                'citation_faithfulness': quality.get("citation_faithfulness", 0.0),
                'disclaimer': disclaimer,
                'query': search_request.text,
                'expanded_query': expanded_query,
                'query_expanded': use_query_expansion,
            }
        )

    return JSONResponse(
        content={
            "signal": Response.RAG_ANSWER_SUCCEED.value,
            'answer': answer,
            'sources': sources,
            'full_prompt': full_prompt,
            'chat_history': chat_history,
            'retrieval_mode': search_request.retrieval_mode,
            'rerank': use_rerank,
            'risk_assessment': risk_assessment,
            'confidence': confidence,
            'confidence_summary': NLPController._build_confidence_summary(confidence),
            'evidence_panel': evidence_panel,
            'pipeline_metadata': pipeline_metadata,
            'quality': quality,
            'citations': quality.get("citations", []),
            'unsupported_claims': quality.get("unsupported_claims", []),
            'unsupported_claim_rate': quality.get("unsupported_claim_rate", 0.0),
            'citation_faithfulness': quality.get("citation_faithfulness", 0.0),
            'disclaimer': disclaimer,
            'query': search_request.text,
            'expanded_query': expanded_query,
            'query_expanded': use_query_expansion,
        }
    )


@nlp_router.post('/index/answer/{project_id}/stream')
async def answer_index_info_stream(request: Request, project_id: int, search_request: SearchRequest):

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
        template_parser=request.app.template_parser,
        rerank_client=request.app.rerank_client
    )

    use_rerank = search_request.rerank if search_request.rerank is not None else True
    use_query_expansion = search_request.expand_query if search_request.expand_query is not None else True
    conversation_history = [turn.model_dump() for turn in (search_request.conversation_history or [])]

    async def event_generator():
        async for event in nlp_controller.answer_rag_question_stream(
            project=project,
            query=search_request.text,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            metadata_filter=search_request.metadata_filter,
            include_sources=search_request.include_sources,
            retrieval_mode=search_request.retrieval_mode,
            rerank=use_rerank,
            expand_query=use_query_expansion,
            verify_claims=bool(search_request.verify_claims),
            conversation_history=conversation_history,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
