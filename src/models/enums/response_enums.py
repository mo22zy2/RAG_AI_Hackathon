from enum import Enum

class Response(Enum):
    
    FILE_VALIDATED_SUCCESS="File validate successfully"
    FILE_TYPE_NOT_SUPPORTED="File Type is Not supported"
    FILE_SIZE_EXCEEDED="File Size is Bigger than 16Mb"
    FILE_UPLOAD_SUCCED="File upload Succeed"
    FILE_VALIDATED_FALIED="File validate Failed"
    FILE_PROCESSING_FALIED="File Processing Failed"
    FILE_PROCESSING_SUCCEED="File Processing SUCESS"
    PROCESSING_FAILED="File Processing Failed"
    NO_FILES_ERROR="Not Found Files"
    FILE_ID_ERROR="No File Record found with this ID"
    PROJECT_NOT_FOUND_ERROR="project_was_not_found"
    INSERT_INTO_VECTOR_DB_ERROR="Error while inserting in vector DB"
    INSERT_INTO_VECTOR_DB_SUCCESS="Done inserting into Vector DB"
    VECTORDB_COLLECTION_RETREIVED="Vector DB Collection Retrived"
    SEARCH_IN_VECTOR_DB_ERROR="Error while searching in Vector DB"
    SEARCH_IN_VECTOR_DB_SUCCESS="Done searching in Vector DB"
    RAG_ANSWER_ERROR="Error while generating The RAG Answer"
    RAG_ANSWER_SUCCEED="Done generating The RAG Answer"
    RAG_NO_DOCUMENTS_FOUND="No relevant documents found to answer the question"
    
    
