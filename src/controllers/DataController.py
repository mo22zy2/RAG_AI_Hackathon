from .BaseController import BaseController
from fastapi import UploadFile
from models import Response
from .ProjectController import ProjectController
import re ,os

PDF_MAGIC_BYTES = b"%PDF-"
CONTENT_SNIFF_BYTES = 1024

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale=1048576

    async def validate_upload_file(self,file):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False,Response.FILE_TYPE_NOT_SUPPORTED.value
        if file.size is not None and file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False,Response.FILE_SIZE_EXCEEDED.value
        if not await self._content_matches_declared_type(file):
            return False,Response.FILE_TYPE_NOT_SUPPORTED.value
        return True ,Response.FILE_UPLOAD_SUCCED.value

    @staticmethod
    async def _content_matches_declared_type(file):
        """
        The client controls `content_type`, so it can't be trusted on its
        own — a non-PDF payload can be labelled `application/pdf`. Sniff the
        actual bytes instead, then rewind so the caller's own read (upload
        streaming, or ProcessController's loader) still sees the full file
        from the start.
        """
        head = await file.read(CONTENT_SNIFF_BYTES)
        await file.seek(0)

        if file.content_type == "application/pdf":
            return head.startswith(PDF_MAGIC_BYTES)
        if file.content_type == "text/plain":
            try:
                head.decode("utf-8")
            except UnicodeDecodeError:
                return False
        return True
    
    
    
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
        
        
        