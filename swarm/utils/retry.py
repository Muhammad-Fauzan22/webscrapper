#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTELLIGENT RETRY MECHANISM
Version: 3.0.0
Created: 2025-07-17
Author: Resilience Engineering Team
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
        self.error_history = []
    
    async def execute(
        self,
        func: Callable,
        *args,
        token_budget_remaining: int = 0,
        **kwargs
    ) -> Any:
        """Execute function with adaptive retry logic"""
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                # Check token budget
                estimated_cost = getattr(func, 'estimated_token_cost', 0)
                
                if token_budget_remaining < estimated_cost:
                    logger.warning(
                        f"Insufficient token budget for operation: "
                        f"{token_budget_remaining} < {estimated_cost}"
                    )
                    raise RuntimeError("Token budget exceeded")
                
                # Execute function
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                attempt += 1
                error_type = type(e).__name__
                
                # Record error
                self.error_history.append({
                    "timestamp": asyncio.get_event_loop().time(),
                    "error_type": error_type,
                    "message": str(e),
                    "attempt": attempt
                })
                
                # Don't retry on certain errors
                if "invalid" in str(e).lower() or "unsupported" in str(e).lower():
                    logger.error(f"Non-retriable error: {str(e)}")
                    break
                
                # Calculate adaptive backoff
                backoff = self._calculate_backoff(attempt, error_type)
                
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed. "
                    f"Retrying in {backoff:.2f}s. Error: {str(e)}"
                )
                
                await asyncio.sleep(backoff)
        
        logger.error(f"All retry attempts failed for {func.__name__}")
        raise last_error or RuntimeError("Unknown error in retry handler")
    
    def _calculate_backoff(self, attempt: int, error_type: str) -> float:
        """Calculate backoff with adaptive strategy"""
        base_backoff = self.backoff_factor * (2 ** attempt)
        
        # Adjust based on error type
        if "RateLimit" in error_type:
            # Longer backoff for rate limits
            multiplier = 2.0
        elif "Connection" in error_type:
            # Moderate backoff for connection issues
            multiplier = 1.5
        else:
            # Standard backoff
            multiplier = 1.0
        
        # Add jitter
        jitter = random.uniform(0.8, 1.2)
        
        return base_backoff * multiplier * jitter
    
    def get_error_stats(self) -> dict:
        """Get statistics on recent errors"""
        if not self.error_history:
            return {}
            
        error_counts = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "last_error": self.error_history[-1] if self.error_history else None
        }
