#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONGODB STORAGE MODULE
Version: 2.0.2
Created: 2025-07-15
Author: Database Team
"""

import os
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger("MongoStorage")

class MongoDBStorage:
    def __init__(self, db_name="ai_scraper", collection_name="scraped_data"):
        self.client = None
        self.db = None
        self.collection = None
        self.db_name = db_name
        self.collection_name = collection_name
        self.connect()
        
    def connect(self):
        """Connect to MongoDB with robust error handling"""
        try:
            # Get connection string from environment
            conn_str = os.getenv("MONGO_CONN_STR")
            if not conn_str:
                raise ValueError("MongoDB connection string not set")
                
            self.client = MongoClient(
                conn_str,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000,
                connectTimeoutMS=10000
            )
            
            # Verify connection
            self.client.server_info()
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Create indexes
            self.collection.create_index("metadata.content_hash")
            self.collection.create_index([("metadata.scrape_time", -1)])
            
            logger.info("MongoDB connection established")
        except PyMongoError as e:
            logger.critical(f"MongoDB connection failed: {str(e)}")
            self.client = None
            raise

    async def save_batch(self, data: list) -> int:
        """Save data batch to MongoDB"""
        if not data or not self.client:
            return 0
            
        try:
            # Add MongoDB-specific metadata
            processed = []
            for item in data:
                item['_insert_time'] = datetime.now()
                item['_batch_size'] = len(data)
                processed.append(item)
                
            result = self.collection.insert_many(processed)
            count = len(result.inserted_ids)
            logger.info(f"Inserted {count} documents into MongoDB")
            return count
        except PyMongoError as e:
            logger.error(f"MongoDB insert failed: {str(e)}")
            return 0

    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
