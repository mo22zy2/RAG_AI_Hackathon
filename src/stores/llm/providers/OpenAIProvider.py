from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
from typing import Union, List
import logging

class OpenAIProvider(LLMInterface):
    
    def __init__(self,api_key:str ,base_url:str=None,
                 default_max_input_chars:int=1000,
                 default_max_output_tokens:int=1000,
                 default_temperature:float=0.8):
        
        self.api_key = api_key
        self.base_url = base_url
        self.default_max_input_chars = default_max_input_chars
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature
        self.generation_model_id=None
        self.embedding_model_id=None
        self.embedding_size=None
        
        self.enums=OpenAIEnums
        self.client=OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.logger = logging.getLogger(__name__) #Get a logger for this class (specific to this module)
         
    def set_generation_model(self, model_id:str): # making able to change the model during runtime
        self.generation_model_id=model_id
        
    def set_embedding_model(self, model_id:str, embedding_size:int): # making able to change the model during runtime
        self.embedding_model_id=model_id
        self.embedding_size=embedding_size
        
            
    def process_text(self,text:str):
        
        if len(text) > self.default_max_input_chars:
            
            self.logger.warning(f"Input text exceeds the maximum allowed characters ({self.default_max_input_chars}). It will be truncated.")
            
            text = text[:self.default_max_input_chars].strip()  # Truncate and remove leading/trailing whitespace
        
        return text
        
    def generate_text(self, prompt:str,
                    chat_history:list=None,
                    max_output_tokens:int=None,
                    temperature:float=0.8):
        
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set. Please set it using set_generation_model().")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_max_output_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        
        messages = (chat_history or []).copy()
        messages.append(
            self.construct_prompt(prompt, OpenAIEnums.USER.value)
        )
        
        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
        
        if not response or not response.choices or len(response.choices) == 0 or response.choices[0].message is None:
            self.logger.error("No response returned from OpenAI API.")
            return None
        
        return response.choices[0].message.content

    def embed_text(self, text:Union[str,List[str]], document_type:str=None):
        
        if not self.client:
            
            self.logger.error("OpenAI client is not initialized.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set. Please set it using set_embedding_model().")
            return None
        
        if isinstance(text,str):
            text=[text]
        
        response = self.client.embeddings.create(model=self.embedding_model_id, 
                                                    input=text)
        
        if not response or not response.data or len(response.data) == 0:
            self.logger.error("No embedding returned from OpenAI API.")
            return None
        
        return [rec.embedding for rec in response.data]
    
    
    def construct_prompt(self,prompt:str,role:str):
        return {
            "role":role,
            "content":prompt
        }