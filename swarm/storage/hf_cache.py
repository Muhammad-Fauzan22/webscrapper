#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HUGGING FACE CACHE MANAGER
Version: 2.0.0
Created: 2025-07-17
Author: Performance Optimization Team
"""

import os
import logging
import hashlib
import json
from datetime import datetime, timedelta
from swarm.utils import Logger

logger = Logger(name="HFCache")

class HFCacheManager:
    def __init__(self, cache_dir: str = "hf_cache", max_size: int = 1024):
        self.cache_dir = cache_dir
        self.max_size = max_size  # in MB
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key: str, max_age_hours: int = 24) -> dict:
        """Get cached item if exists and not expired"""
        filepath = self._key_to_path(key)
        if not os.path.exists(filepath):
            return None
        
        # Check age
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
        if file_age > timedelta(hours=max_age_hours):
            os.remove(filepath)
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Cache read failed: {str(e)}")
            return None
    
    def set(self, key: str, data: dict, ttl_hours: int = 24) -> bool:
        """Store data in cache"""
        try:
            filepath = self._key_to_path(key)
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            # Apply cache eviction if needed
            if self._cache_size() > self.max_size:
                self._evict_oldest(0.2)  # Remove 20% oldest items
            
            return True
        except Exception as e:
            logger.error(f"Cache write failed: {str(e)}")
            return False
    
    def clear(self, max_age_hours: int = None) -> int:
        """Clear cache, optionally by age"""
        removed = 0
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if max_age_hours:
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_age < timedelta(hours=max_age_hours):
                    continue
            
            try:
                os.remove(filepath)
                removed += 1
            except:
                pass
        
        logger.info(f"Cleared {removed} cache items")
        return removed
    
    def _key_to_path(self, key: str) -> str:
        """Convert key to filesystem path"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def _cache_size(self) -> int:
        """Get current cache size in MB"""
        total_size = 0
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _evict_oldest(self, fraction: float = 0.1) -> int:
        """Evict oldest cache items"""
        files = []
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            mtime = os.path.getmtime(filepath)
            files.append((filepath, mtime))
        
        # Sort by oldest first
        files.sort(key=lambda x: x[1])
        
        # Remove fraction of oldest files
        remove_count = int(len(files) * fraction)
        for i in range(remove_count):
            os.remove(files[i][0])
        
        logger.info(f"Evicted {remove_count} oldest cache items")
        return remove_count
