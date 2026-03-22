import os
import shutil
from fastapi import UploadFile

class LocalStorageService:
    def __init__(self):
        self.upload_dir = "./local_storage"
        # Ensure the directory exists locally
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    async def upload_file(self, file_content: bytes, file_name: str):
        """Saves the file to the local disk instead of R2 [cite: 261]"""
        file_path = os.path.join(self.upload_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path

    def get_local_url(self, file_name: str):
        """Returns a local static URL for the frontend to display the file"""
        return f"http://localhost:8000/static/{file_name}"


# import boto3
# from botocore.config import Config
# from app.core.config import settings

# class R2Service:
#     def __init__(self):
#         self.s3_client = boto3.client(
#             service_name='s3',
#             endpoint_url=f'https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
#             aws_access_key_id=settings.R2_ACCESS_KEY,
#             aws_secret_access_key=settings.R2_SECRET_KEY,
#             region_name="auto",
#             config=Config(signature_version='s3v4')
#         )

#     async def upload_file(self, file_content: bytes, file_name: str, content_type: str):
#         """Uploads a document to R2 and returns the object key."""
#         self.s3_client.put_object(
#             Bucket=settings.R2_BUCKET_NAME,
#             Key=file_name,
#             Body=file_content,
#             ContentType=content_type
#         )
#         return file_name

#     def get_signed_url(self, file_key: str, expires_in: int = 3600):
#         """Generates a temporary secure URL for viewing the document."""
#         return self.s3_client.generate_presigned_url(
#             'get_object',
#             Params={'Bucket': settings.R2_BUCKET_NAME, 'Key': file_key},
#             ExpiresIn=expires_in
#         )