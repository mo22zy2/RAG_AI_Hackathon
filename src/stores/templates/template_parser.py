from importlib import import_module
import os
class Template_Parser:
    
    def __init__(self,language:str=None,default_language="en"):
        
        self.current_path=os.path.dirname(os.path.abspath(__file__))
        self.default_language=default_language
        self.set_language(language)
        
        
    def set_language(self,language:str):
        
        if not language:
            self.language=self.default_language
            return
        
        language_path=os.path.join(self.current_path,"locales",language)
        if os.path.exists(language_path):
            self.language=language
        
        else:
            self.language=self.default_language
            
    def get(self, group: str, key: str, variables=None):
        if variables is None:
            variables = {}
        
        target_language = self.language

        group_path = os.path.join(
            self.current_path,
            "locales",
            target_language,
            f"{group}.py"
        )

        if not os.path.exists(group_path):
            target_language = self.default_language
        
        
        module = import_module(
    f"stores.templates.locales.{target_language}.{group}"
)
        
        key_attribute = getattr(module, key, None)

        if key_attribute is None:
            return None
        
        return key_attribute.safe_substitute(variables)
        
            
            
        