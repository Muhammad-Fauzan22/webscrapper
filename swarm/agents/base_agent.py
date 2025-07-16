#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASE AGENT CLASS
Version: 1.0.0
Created: 2025-07-17
Author: Core Architecture Team
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from swarm.utils import CostOptimizer, Logger

logger = Logger(name="BaseAgent")

class BaseAgent(ABC):
    def __init__(self, primary_config: dict, fallback_config: dict, cost_optimizer: CostOptimizer):
        self.primary_config = primary_config
        self.fallback_config = fallback_config
        self.cost_optimizer = cost_optimizer
        self.current_config = primary_config
        self.use_fallback = False
        self.token_usage = 0
        self.retry_handler = self._create_retry_handler()
    
    def _create_retry_handler(self):
        """Create retry handler with agent-specific settings"""
        return RetryHandler(
            max_retries=3,
            backoff_factor=1.5,
            retriable_errors=["ConnectionError", "TimeoutError", "RateLimitError"]
        )
    
    def switch_to_fallback(self):
        """Switch to fallback configuration"""
        if not self.use_fallback:
            logger.warning(f"Switching to fallback provider for {self.__class__.__name__}")
            self.current_config = self.fallback_config
            self.use_fallback = True
    
    def switch_to_primary(self):
        """Switch back to primary configuration"""
        if self.use_fallback:
            logger.info(f"Reverting to primary provider for {self.__class__.__name__}")
            self.current_config = self.primary_config
            self.use_fallback = False
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        return await self.retry_handler.execute(
            func,
            *args,
            token_budget_remaining=self.cost_optimizer.get_remaining_budget(),
            **kwargs
        )
    
    def _update_token_usage(self, tokens: int):
        """Update token usage and track cost"""
        if tokens <= 0:
            return
        
        self.token_usage += tokens
        provider = "fallback" if self.use_fallback else "primary"
        self.cost_optimizer.track_usage(provider, tokens)
        
        # Switch to fallback if approaching budget
        if not self.use_fallback and self.cost_optimizer.should_switch(provider):
            self.switch_to_fallback()
    
    @abstractmethod
    async def execute(self, *args, **kwargs):
        """Main execution method to be implemented by subclasses"""
        pass
    
    def get_status(self) -> dict:
        """Get current agent status"""
        return {
            "agent": self.__class__.__name__,
            "provider": "fallback" if self.use_fallback else "primary",
            "token_usage": self.token_usage,
            "last_activity": datetime.now().isoformat()
        }
