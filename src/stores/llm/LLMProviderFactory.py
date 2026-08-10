from .LLMEnums import LLMType
from .providers import OpenAIProvider, CoHereProvider


class LLMProviderFactory:
    
    def __init__(self,config:dict):
        self.config=config
        
    def create_provider(self,provider:str):
        
        if provider == LLMType.OPENAI.value:
            
            
            return OpenAIProvider(
    api_key=self.config.OPENAI_API_KEY,
    base_url=self.config.OPENAI_BASE_URL,
    default_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
    default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
    default_max_input_chars=self.config.INPUT_DEFAULT_MAX_CHARS,
)
        
        
        
        if provider == LLMType.COHERE.value:
            return CoHereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
                default_max_input_chars=self.config.INPUT_DEFAULT_MAX_CHARS,
)
            
        
        return None
        