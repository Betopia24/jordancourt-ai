import os
import boto3
import logging
from typing import Optional, Tuple
from dotenv import load_dotenv
import mimetypes
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.region = os.getenv("AWS_REGION")
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.public_read = os.getenv("AWS_S3_PUBLIC_READ", "true").strip().lower() == "true"
        self.object_prefix = self._normalize_prefix(os.getenv("AWS_S3_OBJECT_PREFIX", "chat_uploads"))
        
        # Validate required environment variables
        if not all([self.region, self.access_key, self.secret_key, self.bucket_name]):
            missing = [k for k, v in {
                "AWS_REGION": self.region,
                "AWS_ACCESS_KEY_ID": self.access_key,
                "AWS_SECRET_ACCESS_KEY": self.secret_key,
                "AWS_S3_BUCKET_NAME": self.bucket_name
            }.items() if not v]
            raise ValueError(f"Missing required AWS environment variables: {', '.join(missing)}")
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=boto3.session.Config(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    connect_timeout=5,
                    read_timeout=10
                )
            )
            # Verify bucket access on initialization
            self._verify_bucket_access()
            self._ensure_storage_folder()
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _normalize_prefix(self, prefix: str) -> str:
        """Normalize and sanitize the S3 key prefix used as a logical folder."""
        cleaned = (prefix or "").strip().strip('/')
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c in '-_/')
        while '//' in cleaned:
            cleaned = cleaned.replace('//', '/')

        if not cleaned:
            cleaned = "chat_uploads"

        return f"{cleaned}/"
    
    def _verify_bucket_access(self):
        """Verify that the bucket exists and is accessible"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket access verified: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                raise ValueError(f"S3 bucket does not exist: {self.bucket_name}")
            elif error_code == '403':
                raise ValueError(f"Access denied to S3 bucket: {self.bucket_name}")
            else:
                raise ValueError(f"Cannot access S3 bucket: {error_code}")

    def _ensure_storage_folder(self):
        """Create a folder marker object so the S3 prefix is visible as a folder."""
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=self.object_prefix, Body=b"")
            logger.info(f"S3 storage folder ready: {self.object_prefix}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            raise ValueError(f"Unable to create S3 storage folder '{self.object_prefix}': {error_code}")
    
    def _detect_image_format(self, image_data: bytes, filename: str = None) -> Tuple[str, str]:
        """
        Detect image format from file signature (magic bytes) and filename
        Returns (extension, content_type)
        """
        # Validate input
        if not image_data or len(image_data) < 20:
            logger.warning("Image data too small for format detection")
            return 'jpg', 'image/jpeg'
        
        # Check magic bytes (file signatures)
        if image_data.startswith(b'\xff\xd8\xff'):
            return 'jpg', 'image/jpeg'
        elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png', 'image/png'
        elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
            return 'gif', 'image/gif'
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
            return 'webp', 'image/webp'
        elif image_data.startswith(b'BM'):
            return 'bmp', 'image/bmp'
        elif image_data.startswith(b'\x00\x00\x01\x00'):
            return 'ico', 'image/x-icon'
        
        # Fallback to filename extension
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else None
            if ext and ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'ico']:
                content_type = mimetypes.guess_type(filename)[0] or 'image/jpeg'
                return ext, content_type
        
        # Default fallback
        logger.warning(f"Could not detect image format, using default (jpg)")
        return 'jpg', 'image/jpeg'
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        reraise=True
    )
    def upload_image(self, image_data: bytes, user_id: str, chat_id: str, image_count: int, filename: str = None) -> Optional[str]:
        """
        Upload image to S3 and return the public URL
        
        Args:
            image_data: Raw image bytes
            user_id: User ID
            chat_id: Chat ID
            image_count: Sequence number of the image in this chat
            filename: Original filename for format detection
            
        Returns:
            S3 URL of the uploaded image or None if failed
        """
        # Validate inputs
        if not image_data:
            logger.error("Cannot upload empty image data")
            return None
        
        if not all([user_id, chat_id]) or image_count < 1:
            logger.error(f"Invalid parameters: user_id={user_id}, chat_id={chat_id}, image_count={image_count}")
            return None
        
        # Check image size (max 10MB for safety)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            logger.error(f"Image size ({len(image_data)} bytes) exceeds maximum ({max_size} bytes)")
            return None
        
        try:
            # Detect image format
            extension, content_type = self._detect_image_format(image_data, filename)
            
            # Sanitize user inputs for S3 key
            safe_user_id = ''.join(c for c in user_id if c.isalnum() or c in '-_')[:50]
            safe_chat_id = ''.join(c for c in chat_id if c.isalnum() or c in '-_')[:50]
            
            # Generate S3 key with correct extension
            s3_key = f"{self.object_prefix}{safe_user_id}_{safe_chat_id}_{image_count}.{extension}"

            # Upload to S3 and set object-level public access by default.
            put_kwargs = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': image_data,
                'ContentType': content_type,
                'ServerSideEncryption': 'AES256',  # Enable encryption at rest
                'Metadata': {
                    'user_id': safe_user_id,
                    'chat_id': safe_chat_id,
                    'image_count': str(image_count)
                }
            }
            if self.public_read:
                put_kwargs['ACL'] = 'public-read'

            try:
                self.s3_client.put_object(**put_kwargs)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if self.public_read and error_code in ('AccessControlListNotSupported', 'InvalidRequest'):
                    # Buckets with Object Ownership "Bucket owner enforced" reject ACLs.
                    # Retry without ACL so upload succeeds; bucket policy must allow public reads.
                    logger.warning(
                        "Bucket does not allow ACLs (%s). Retrying upload without ACL. "
                        "Configure bucket policy for public read if public URLs are required.",
                        error_code
                    )
                    put_kwargs.pop('ACL', None)
                    self.s3_client.put_object(**put_kwargs)
                else:
                    raise
            
            # Generate public URL
            image_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(
                f"Image uploaded to S3: {s3_key} (format: {extension}, size: {len(image_data)} bytes, public_read={self.public_read})"
            )
            return image_url
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"S3 ClientError uploading image: {error_code} - {str(e)}")
            return None
        except BotoCoreError as e:
            logger.error(f"BotoCoreError uploading image: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading image to S3: {e}", exc_info=True)
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        reraise=True
    )
    def delete_image(self, user_id: str, chat_id: str, image_count: int, extension: str = 'jpg') -> bool:
        """Delete an image from S3"""
        if not all([user_id, chat_id]) or image_count < 1:
            logger.error(f"Invalid parameters for deletion: user_id={user_id}, chat_id={chat_id}, image_count={image_count}")
            return False
        
        try:
            # Sanitize inputs
            safe_user_id = ''.join(c for c in user_id if c.isalnum() or c in '-_')[:50]
            safe_chat_id = ''.join(c for c in chat_id if c.isalnum() or c in '-_')[:50]
            safe_extension = ''.join(c for c in extension if c.isalnum())[:10]
            
            s3_key = f"{self.object_prefix}{safe_user_id}_{safe_chat_id}_{image_count}.{safe_extension}"
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Image deleted from S3: {s3_key}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"S3 ClientError deleting image: {error_code} - {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting image from S3: {e}", exc_info=True)
            return False

# Singleton instance
try:
    s3_manager = S3Manager()
except Exception as e:
    logger.critical(f"Failed to initialize S3Manager singleton: {e}")
    raise
