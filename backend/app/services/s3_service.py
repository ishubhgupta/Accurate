import boto3
from flask import current_app
from botocore.exceptions import ClientError
import io

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=current_app.config.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=current_app.config.get('AWS_SECRET_ACCESS_KEY'),
            region_name=current_app.config.get('AWS_REGION')
        )
        self.bucket_name = current_app.config.get('AWS_S3_BUCKET')
    
    def upload_file(self, file_obj, s3_path):
        """Upload file to S3"""
        try:
            # If it's a file-like object from Flask, read the content
            if hasattr(file_obj, 'read'):
                file_content = file_obj.read()
                file_obj = io.BytesIO(file_content)
            
            self.s3_client.upload_fileobj(
                file_obj, 
                self.bucket_name, 
                s3_path,
                ExtraArgs={'ServerSideEncryption': 'AES256'}
            )
            return True
        except ClientError as e:
            current_app.logger.error(f"S3 upload error: {str(e)}")
            return False
    
    def download_file(self, s3_path):
        """Download file from S3"""
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_path)
            return obj['Body'].read()
        except ClientError as e:
            current_app.logger.error(f"S3 download error: {str(e)}")
            return None
    
    def delete_file(self, s3_path):
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_path)
            return True
        except ClientError as e:
            current_app.logger.error(f"S3 delete error: {str(e)}")
            return False
    
    def get_file_url(self, s3_path, expiration=3600):
        """Generate presigned URL for file access"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_path},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            current_app.logger.error(f"S3 URL generation error: {str(e)}")
            return None