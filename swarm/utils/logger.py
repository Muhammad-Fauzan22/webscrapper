#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED LOGGING SYSTEM
Version: 2.0.0
Created: 2025-07-17
Author: Observability Team
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pymongo import MongoClient

class Logger:
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (daily rotation)
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            f"{log_dir}/{name}.log",
            when="midnight",
            backupCount=7
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # MongoDB logging if configured
        if os.getenv("MONGO_URI"):
            self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
            self.db = self.mongo_client[os.getenv("MONGO_DB_NAME", "logs_db")]
            self.logs_collection = self.db["system_logs"]
        else:
            self.mongo_client = None
    
    def log(self, level: str, message: str, **kwargs):
        """Log message with additional context"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "level": level,
            "message": message,
            "context": kwargs
        }
        
        # Log to standard handlers
        getattr(self.logger, level.lower())(message, extra=kwargs)
        
        # Log to MongoDB if available
        if self.mongo_client:
            try:
                self.logs_collection.insert_one(log_entry)
            except Exception as e:
                print(f"Failed to log to MongoDB: {str(e)}")
    
    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log("DEBUG", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log("CRITICAL", message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra=kwargs)
        if self.mongo_client:
            log_entry = {
                "timestamp": datetime.utcnow(),
                "level": "EXCEPTION",
                "message": message,
                "context": kwargs,
                "exception": True
            }
            self.logs_collection.insert_one(log_entry)
