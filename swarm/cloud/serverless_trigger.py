#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERVERLESS FUNCTION TRIGGER
Version: 2.0.0
Created: 2025-07-17
Author: Cloud Integration Team
"""

import os
import logging
import requests
from swarm.utils import Logger

logger = Logger(name="ServerlessTrigger")

class ServerlessTrigger:
    def __init__(self):
        self.github_token = os.getenv("GH_TOKEN")
        self.repo = os.getenv("GH_REPO")
        self.azure_function_url = os.getenv("AZURE_FUNCTION_URL")
    
    async def trigger_restart(self) -> bool:
        """Trigger a system restart via serverless function"""
        logger.warning("Initiating system restart via serverless function")
        
        # First try GitHub Actions
        if self.github_token and self.repo:
            if await self._trigger_github_action():
                return True
        
        # Fallback to Azure Functions
        if self.azure_function_url:
            return await self._trigger_azure_function()
        
        logger.error("No restart method available")
        return False
    
    async def _trigger_github_action(self) -> bool:
        """Trigger GitHub Actions workflow"""
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            payload = {
                "ref": "main",
                "inputs": {"action": "restart"}
            }
            
            response = requests.post(
                f"https://api.github.com/repos/{self.repo}/actions/workflows/restart.yml/dispatches",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("GitHub Actions restart triggered")
                return True
            else:
                logger.error(f"GitHub trigger failed: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"GitHub trigger exception: {str(e)}")
            return False
    
    async def _trigger_azure_function(self) -> bool:
        """Trigger Azure Function"""
        try:
            response = requests.post(
                self.azure_function_url,
                json={"action": "restart"},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Azure Function restart triggered")
                return True
            else:
                logger.error(f"Azure Function trigger failed: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Azure Function trigger exception: {str(e)}")
            return False
    
    async def trigger_retrain(self) -> bool:
        """Trigger model retraining"""
        logger.info("Triggering model retraining")
        # Similar implementation to restart
        return await self._trigger_github_action("retrain.yml")
