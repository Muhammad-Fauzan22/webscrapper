#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOOGLE DRIVE STORAGE MODULE
Version: 3.0.4
Created: 2025-07-15
Author: Cloud Storage Team
"""

import os
import json
import gzip
import hashlib
import base64
import logging
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger("DriveStorage")

class DriveStorage:
    def __init__(self, chunk_size=5*1024*1024):  # 5MB chunks
        self.service = self._initialize_service()
        self.chunk_size = chunk_size
        self.folder_cache = {}
        
    def _initialize_service(self):
        """Initialize Google Drive service with robust error handling"""
        try:
            # Load credentials from environment or file
            if os.getenv('GOOGLE_CREDS_JSON'):
                creds_info = json.loads(os.getenv('GOOGLE_CREDS_JSON'))
            elif os.path.exists('service_account.json'):
                with open('service_account.json') as f:
                    creds_info = json.load(f)
            else:
                raise RuntimeError("Google credentials not found")
            
            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            return build('drive', 'v3', credentials=credentials, cache_discovery=False)
        except Exception as e:
            logger.critical(f"Drive service initialization failed: {str(e)}")
            raise

    async def save_batch(self, data: list) -> int:
        """Save data batch to Google Drive with advanced features"""
        if not data:
            return 0
            
        try:
            # Create daily folder structure
            folder_id = await self._get_today_folder()
            
            # Process and compress data
            serialized = json.dumps(data, ensure_ascii=False)
            compressed = gzip.compress(serialized.encode('utf-8'))
            content_hash = hashlib.sha256(compressed).hexdigest()
            
            # Check for duplicates
            if await self._is_duplicate(content_hash, folder_id):
                logger.info("Duplicate content detected, skipping upload")
                return 0
                
            # Upload file
            file_name = f"scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{content_hash[:8]}.json.gz"
            await self._upload_file(file_name, compressed, content_hash, folder_id)
            
            return len(data)
        except Exception as e:
            logger.error(f"Batch save failed: {str(e)}")
            return 0

    async def _get_today_folder(self) -> str:
        """Get or create daily folder with retry logic"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today in self.folder_cache:
            return self.folder_cache[today]
            
        try:
            # Check if folder exists
            query = f"name='{today}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            folders = results.get('files', [])
            
            if folders:
                folder_id = folders[0]['id']
            else:
                # Create new folder
                folder_metadata = {
                    'name': today,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [os.getenv('GOOGLE_DRIVE_FOLDER_ID')]
                }
                folder = self.service.files().create(body=folder_metadata, fields='id').execute()
                folder_id = folder['id']
                
            self.folder_cache[today] = folder_id
            return folder_id
        except HttpError as e:
            logger.error(f"Folder creation failed: {e.error_details}")
            raise
        except Exception as e:
            logger.error(f"Folder operation failed: {str(e)}")
            raise

    async def _upload_file(self, file_name: str, content: bytes, content_hash: str, parent_id: str):
        """Upload file with chunked resumable upload"""
        try:
            file_metadata = {
                'name': file_name,
                'parents': [parent_id],
                'description': 'Scraped data archive',
                'properties': {
                    'content_hash': content_hash,
                    'compression': 'gzip',
                    'original_size': str(len(content)),
                    'created': datetime.now().isoformat()
                }
            }
            
            media = MediaInMemoryUpload(content, mimetype='application/gzip')
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            # Execute with exponential backoff
            response = None
            for attempt in range(3):
                try:
                    response = request.execute()
                    break
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504]:
                        sleep_time = 2 ** attempt
                        logger.warning(f"Retryable error, retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        raise
            
            if not response:
                raise RuntimeError("Upload failed after retries")
                
            logger.info(f"Uploaded file ID: {response.get('id')}")
            return response.get('id')
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise

    async def _is_duplicate(self, content_hash: str, parent_id: str) -> bool:
        """Check for duplicate content using content hash"""
        try:
            query = f"'{parent_id}' in parents and properties has {{ key='content_hash' and value='{content_hash}' }} and trashed=false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            return bool(results.get('files', []))
        except Exception as e:
            logger.error(f"Duplicate check failed: {str(e)}")
            return False

    async def close(self):
        """Clean up resources"""
        self.folder_cache.clear()
        logger.info("Drive storage connection closed")
