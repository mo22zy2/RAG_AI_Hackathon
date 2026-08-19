from abc import ABC, abstractmethod

class LLMInterface(ABC):
    
    
    @abstractmethod
    def set_generation_model(self,model_id:str):
        pass
    
    
    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass
    
    @abstractmethod
    async def generate_text(self, prompt:str,
                      chat_history:list=None,
                      max_output_tokens:int=None,
                      temperature:float=0.8):
        pass

    async def generate_text_stream(self, prompt:str,
                             chat_history:list=None,
                             max_output_tokens:int=None,
                             temperature:float=0.8):
        raise NotImplementedError("Streaming not implemented for this provider")


    @abstractmethod
    async def embed_text(self,text:str,document_type:str=None):
        pass
    
    
    @abstractmethod
    def construct_prompt(self,prompt:str,role:str):
        pass
    
    
    
    