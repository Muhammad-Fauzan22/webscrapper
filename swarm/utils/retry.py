#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED RETRY MECHANISM
Version: 2.0.0
Created: 2025-07-17
Author: System Resilience Team
"""

import asyncio
import random
import logging
from typing import Callable, Any

logger = logging.getLogger("RetryHandler")

class RetryHandler:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute(
        self,
        func: Callable,
        *args,
        token_budget_remaining: int = 0,
        **kwargs
    ) -> Any:
        """Execute function with smart retry logic"""
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                # Estimate token cost before execution
                estimated_cost = getattr(func, 'estimated_token_cost', 0)
                
                # Check if we have enough token budget
                if token_budget_remaining < estimated_cost:
                    logger.warning(
                        f"Insufficient token budget for operation: "
                        f"{token_budget_remaining} < {estimated_cost}"
                    )
                    raise RuntimeError("Token budget exceeded")
                
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                
                # Don't retry on certain errors
                if "invalid" in str(e).lower() or "unsupported" in str(e).lower():
                    logger.error(f"Non-retriable error: {str(e)}")
                    break
                
                # Calculate backoff with jitter
                backoff = self.backoff_factor * (2 ** attempt)
                jitter = random.uniform(0.5, 1.5)
                sleep_time = backoff * jitter
                
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed. "
                    f"Retrying in {sleep_time:.2f}s. Error: {str(e)}"
                )
                
                await asyncio.sleep(sleep_time)
        
        logger.error(f"All retry attempts failed for {func.__name__}")
        raise last_error or RuntimeError("Unknown error in retry handler")

def token_cost_estimator(estimated_cost: int):
    """Decorator to estimate token cost for a function"""
    def decorator(func):
        func.estimated_token_cost = estimated_cost
        return func
    return decorator
