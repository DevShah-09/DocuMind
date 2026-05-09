import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class S3StorageService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION.split(' ')[-1]  # Extract 'eu-north-1' if needed
        )
        self.bucket_name = settings.AWS_S3_BUCKET

    async def upload_file(self, file_content: bytes, file_name: str, content_type: str = "application/octet-stream"):
        """Uploads file bytes to AWS S3."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=file_content,
                ContentType=content_type
            )
            logger.info(f"[S3] Uploaded file: {file_name}")
            return file_name
        except ClientError as e:
            logger.error(f"[S3] Upload failed: {e}")
            raise e

    async def download_file(self, file_name: str) -> bytes:
        """Retrieves file bytes from AWS S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"[S3] Download failed: {e}")
            raise e

    def delete_file(self, file_name: str):
        """Deletes file from AWS S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_name
            )
            logger.info(f"[S3] Deleted file: {file_name}")
            return True
        except ClientError as e:
            logger.error(f"[S3] Delete failed: {e}")
            return False