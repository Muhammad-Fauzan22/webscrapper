#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONGODB STORAGE MANAGER
Version: 2.0.0
Created: 2025-07-17
Author: Database Team
"""

import os
import logging
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from datetime import datetime, timedelta
from bson import ObjectId

logger = logging.getLogger("MongoDBManager")

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        
    async def connect(self, uri: str = None, db_name: str = None):
        """Connect to MongoDB instance"""
        try:
            uri = uri or os.getenv("MONGO_URI")
            db_name = db_name or os.getenv("MONGO_DB_NAME")
            
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            logger.info(f"Connected to MongoDB: {db_name}")
            return True
        except PyMongoError as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    async def insert_document(self, collection: str, document: dict) -> str:
        """Insert a single document"""
        try:
            result = self.db[collection].insert_one(document)
            logger.debug(f"Inserted document ID: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Insert failed: {str(e)}")
            return None
    
    async def bulk_insert(self, collection: str, documents: list) -> dict:
        """Insert multiple documents efficiently"""
        try:
            if not documents:
                logger.warning("No documents to insert")
                return {"inserted_count": 0}
                
            result = self.db[collection].insert_many(documents)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
            return {
                "inserted_count": len(result.inserted_ids),
                "inserted_ids": [str(id) for id in result.inserted_ids]
            }
        except PyMongoError as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            return {"error": str(e)}
    
    async def get_recent_data(self, collection: str, days: int = 7, limit: int = 1000) -> list:
        """Retrieve recent data from the collection"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cursor = self.db[collection].find(
                {"timestamp": {"$gte": cutoff_date}},
                limit=limit
            ).sort("timestamp", ASCENDING)
            
            return list(cursor)
        except PyMongoError as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            return []
    
    async def get_validation_data(self, collection: str, sample_size: int = 100) -> list:
        """Get balanced sample for validation"""
        try:
            # Get distinct classes
            pipeline = [
                {"$group": {"_id": "$label", "count": {"$sum": 1}}}
            ]
            class_counts = list(self.db[collection].aggregate(pipeline))
            
            # Sample from each class
            samples = []
            for cls in class_counts:
                class_samples = list(self.db[collection].aggregate([
                    {"$match": {"label": cls["_id"]}},
                    {"$sample": {"size": min(cls["count"], sample_size // len(class_counts))}}
                ]))
                samples.extend(class_samples)
                
            return samples
        except PyMongoError as e:
            logger.error(f"Validation data retrieval failed: {str(e)}")
            return []
    
    async def create_dump(self) -> str:
        """Create database dump (simplified)"""
        try:
            # In a real implementation, this would use mongodump
            dump_path = f"mongodump_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Simplified: export all collections
            for col_name in self.db.list_collection_names():
                data = list(self.db[col_name].find())
                with open(dump_path, "a") as f:
                    for doc in data:
                        f.write(json.dumps(doc) + "\n")
            
            logger.info(f"Database dump created: {dump_path}")
            return dump_path
        except Exception as e:
            logger.error(f"Dump creation failed: {str(e)}")
            return None
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
