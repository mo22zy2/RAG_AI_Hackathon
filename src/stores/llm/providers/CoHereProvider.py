from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentType
from typing import Union,List
import cohere
import logging


class CoHereProvider(LLMInterface):
    
    
    def __init__(self,api_key:str ,
                 default_max_input_chars:int=1000,
                 default_max_output_tokens:int=1000,
                 default_temperature:float=0.8):
        
        self.api_key = api_key

        self.default_max_input_chars = default_max_input_chars
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature
        self.generation_model_id=None
        self.embedding_model_id=None
        self.embedding_size=None
        self.logger = logging.getLogger(__name__) #Get a logger for this class (specific to this module)
         
        self.enums=CoHereEnums
        self.client = cohere.Client(self.api_key)
         
         
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        
        
    def process_text(self,text:str):
            
            if len(text) > self.default_max_input_chars:
               
                self.logger.warning(f"Input text exceeds the maximum allowed characters ({self.default_max_input_chars}). It will be truncated.")
                
                text = text[:self.default_max_input_chars].strip()  # Truncate and remove leading/trailing whitespace
            
            return text        

    
    def generate_text(self, prompt: str,
                      chat_history: list = None,
                      max_output_tokens: int = None,
                      temperature: float = 0.8):
        
        if not self.client:
                self.logger.error("Cohere client is not initialized.")
                return None
            
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set. Please set it using set_generation_model().")
            return None
        
        # Handle None chat_history
        chat_history = chat_history if chat_history is not None else []
        
        response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history,
                message=self.process_text(prompt),
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_output_tokens if max_output_tokens is not None else self.default_max_output_tokens
            )
        if not response or not response.text:
            self.logger.error("No response returned from Cohere API.")
            return None
            
        return response.text
        
        
    def embed_text(self, text: Union[str , List[str]], document_type: str = None):
        if not self.client:
            self.logger.error("Cohere client is not initialized.")
            return None
        
        if isinstance(text,str):
            text=[text]
            
        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set. Please set it using set_embedding_model().")
            return None
        
        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentType.QUERY:
            input_type = CoHereEnums.QUERY.value
        
        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(t) for t in text],
            input_type=input_type,
            embedding_types=["float"]
)
        
        if not response or not response.embeddings or not response.embeddings.float or len(response.embeddings.float) == 0:
            self.logger.error("No embeddings returned from Cohere API.")
            return None
        
        return response.embeddings.float
        
    def construct_prompt(self, prompt: str, role: str):
        
            # Validate role against CoHereEnums
            valid_roles = [
                CoHereEnums.SYSTEM.value,
                CoHereEnums.USER.value,
                CoHereEnums.ASSISTANT.value
            ]
            
            if role not in valid_roles:
                self.logger.warning(f"Invalid role '{role}'. Using default role 'USER'.")
                role = CoHereEnums.USER.value
            
            return {
                "role": role,
                "text": prompt
            }