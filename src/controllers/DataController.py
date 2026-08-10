from .BaseController import BaseController
from fastapi import UploadFile
from models import Response
from .ProjectController import ProjectController
import re ,os
class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale=1048576
        
    def validate_upload_file(self,file):
        
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False,Response.FILE_TYPE_NOT_SUPPORTED.value
        if file.size is not None and file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False,Response.FILE_SIZE_EXCEEDED.value
        return True ,Response.FILE_UPLOAD_SUCCED.value
    
    
    
    def generate_unique_filepath(self, original_file_name: str, project_id: str):

        random_key = self.generate_random_string().replace(" ", "")
        project_path = ProjectController().get_project_path(
            project_id=project_id
        )

        cleaned_file_name = self.get_clean_file_name(
            original_file_name=original_file_name
        )

        new_file_path = os.path.join(
            project_path,
            random_key + "_" + cleaned_file_name
        )

        while os.path.exists(new_file_path):
            random_key = self.generate_random_string().replace(" ", "")

            new_file_path = os.path.join(
                project_path,
                random_key + "_" + cleaned_file_name
            )

        return new_file_path, random_key + "_" + cleaned_file_name    
        
    def get_clean_file_name(self,original_file_name:str):
        
        cleaned_file_name = re.sub(
    r'[^\w\-.]',
    '',
    original_file_name.strip()
)
        
        cleaned_file_name = cleaned_file_name.replace(" ", "_")
        return cleaned_file_name
        
        
        