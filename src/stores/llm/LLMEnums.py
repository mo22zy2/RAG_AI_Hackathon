from enum import Enum

class LLMType(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    ANTHROPIC = "anthropic"
    
    
class OpenAIEnums(Enum):
    SYSTEM="system"
    USER="user"
    ASSISTANT="assistant"
    
    
class CoHereEnums(Enum):
    SYSTEM="SYSTEM"
    USER="USER"
    ASSISTANT="ASSISTANT"
    
    DOCUMENT="search_document"
    QUERY="search_query"
    
    
class DocumentType(Enum):
    DOCUMENT="document"
    QUERY="query"