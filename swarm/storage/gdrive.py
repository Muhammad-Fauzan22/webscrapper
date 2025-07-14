import os
import gzip
import json
import hashlib
import base64
import logging
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from config import TIMESTAMP, USER
from cryptography.fernet import Fernet

logger = logging.getLogger("DriveStorage")

# Encryption key
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default_key_here").encode()

class DriveStorage:
    def __init__(self):
        self.service = self._setup_client()
        self.root_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        self.shard_cache = {}
        
    def _setup_client(self):
        """Setup Google Drive client"""
        try:
            private_key = os.getenv('GOOGLE_PRIVATE_KEY').replace('\\n', '\n')
            
            credentials = service_account.Credentials.from_service_account_info({
                "type": "service_account",
                "project_id": os.getenv('GOOGLE_PROJECT_ID'),
                "private_key": private_key,
                "client_email": os.getenv('GOOGLE_CLIENT_EMAIL'),
                "token_uri": "https://oauth2.googleapis.com/token"
            }, scopes=['https://www.googleapis.com/auth/drive'])
            
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            logger.error(f"Drive client setup failed: {str(e)}")
            return None
    
    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt data using Fernet symmetric encryption"""
        cipher = Fernet(ENCRYPTION_KEY)
        return cipher.encrypt(data.encode('utf-8'))
    
    def _get_shard_id(self) -> str:
        """Get shard ID based on timestamp"""
        return datetime.utcnow().strftime("%Y-%m-%d-%H")
    
    async def _get_shard_folder(self) -> str:
        """Get or create shard folder"""
        shard_id = self._get_shard_id()
        
        if shard_id in self.shard_cache:
            return self.shard_cache[shard_id]
            
        try:
            # Search for existing folder
            query = f"name='{shard_id}' and '{self.root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            results = self.service.files().list(
                q=query, 
                spaces='drive', 
                fields='files(id)'
            ).execute()
            files = results.get('files', [])
            
            if files:
                folder_id = files[0]['id']
            else:
                # Create new folder
                folder_metadata = {
                    'name': shard_id,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [self.root_folder_id],
                    'properties': {
                        'created_by': USER,
                        'timestamp': TIMESTAMP
                    }
                }
                folder = self.service.files().create(
                    body=folder_metadata, 
                    fields='id'
                ).execute()
                folder_id = folder.get('id')
            
            self.shard_cache[shard_id] = folder_id
            return folder_id
        except Exception as e:
            logger.error(f"Shard folder creation failed: {str(e)}")
            return None

    async def save_batch(self, data_batch: list) -> int:
        """Save batch of scraped data with compression and encryption"""
        if not self.service or not data_batch:
            return 0
            
        try:
            shard_folder_id = await self._get_shard_folder()
            if not shard_folder_id:
                return 0
                
            # 1. Prepare data payload
            payload = {
                "metadata": {
                    "shard": self._get_shard_id(),
                    "user": USER,
                    "timestamp": TIMESTAMP,
                    "item_count": len(data_batch)
                },
                "items": data_batch
            }
            json_data = json.dumps(payload, ensure_ascii=False)
            
            # 2. Encrypt data
            encrypted = self._encrypt_data(json_data)
            
            # 3. Compress with gzip
            compressed = gzip.compress(encrypted)
            
            # 4. Generate content hash
            content_hash = hashlib.sha256(compressed).hexdigest()
            
            # 5. Check for duplicates
            if await self._is_duplicate(content_hash, shard_folder_id):
                return 0
                
            # 6. Upload to Drive
            filename = f"scrape-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{content_hash[:8]}.bin"
            return self._upload_file(compressed, filename, content_hash, shard_folder_id)
        except Exception as e:
            logger.error(f"Batch save failed: {str(e)}")
            return 0

    def _upload_file(self, content: bytes, filename: str, content_hash: str, parent_id: str) -> int:
        """Upload file to Google Drive with metadata"""
        file_metadata = {
            'name': filename,
            'parents': [parent_id],
            'description': 'Scraped data bundle',
            'properties': {
                'content_hash': content_hash,
                'compressed': 'true',
                'encrypted': 'true',
                'original_size': str(len(content)),
                'timestamp': datetime.utcnow().isoformat(),
                'user': USER
            }
        }
        
        media = MediaInMemoryUpload(content, mimetype='application/octet-stream')
        try:
            self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            return 1
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return 0

    async def _is_duplicate(self, content_hash: str, parent_id: str) -> bool:
        """Check for duplicate content using hash"""
        try:
            query = f"'{parent_id}' in parents and properties has {{ key='content_hash' and value='{content_hash}' }}"
            results = self.service.files().list(
                q=query, 
                pageSize=1, 
                fields='files(id)'
            ).execute()
            return bool(results.get('files', []))
        except Exception as e:
            logger.error(f"Duplicate check failed: {str(e)}")
            return False
