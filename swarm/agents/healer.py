#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED SELF-HEALING AGENT
Version: 3.0.0
Created: 2025-07-17
Author: System Resilience Team
"""

import asyncio
import logging
import random
from datetime import datetime
from .base_agent import BaseAgent
from swarm.utils import RetryHandler

logger = logging.getLogger("Healer")

class Healer(BaseAgent):
    def __init__(self, primary_config, fallback_config, cost_optimizer):
        super().__init__(primary_config, fallback_config, cost_optimizer)
        self.error_history = []
        self.last_healing = datetime.now()
        self.healing_strategies = self._load_healing_strategies()
        
    async def diagnose_and_heal(self, error: Exception) -> dict:
        """Diagnose and heal system errors"""
        error_type = type(error).__name__
        error_msg = str(error)
        logger.warning(f"Diagnosing error: {error_type} - {error_msg}")
        
        # Record error
        self.error_history.append({
            "timestamp": datetime.now(),
            "type": error_type,
            "message": error_msg
        })
        
        # Analyze error pattern
        analysis = await self._analyze_errors()
        
        # Select and apply healing strategy
        strategy = self._select_strategy(analysis)
        result = await self._apply_healing(strategy)
        
        return {
            "error_type": error_type,
            "strategy": strategy,
            "result": result,
            "analysis": analysis
        }
    
    async def _analyze_errors(self) -> dict:
        """Analyze error patterns"""
        # Simple frequency analysis
        error_counts = {}
        for error in self.error_history:
            error_type = error["type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Time-based analysis (errors in last hour)
        recent_errors = [e for e in self.error_history 
                        if (datetime.now() - e["timestamp"]).seconds < 3600]
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "recent_errors": len(recent_errors),
            "most_common": max(error_counts, key=error_counts.get) if error_counts else None
        }
    
    def _select_strategy(self, analysis: dict) -> str:
        """Select appropriate healing strategy"""
        # Rate limit errors
        if "RateLimit" in analysis.get("most_common", ""):
            return "rotate_provider"
        
        # Connection errors
        if "Connection" in analysis.get("most_common", ""):
            return "reset_connection"
        
        # High frequency of recent errors
        if analysis["recent_errors"] > 10:
            return "restart_component"
        
        # Default strategy
        return "retry_with_backoff"
    
    async def _apply_healing(self, strategy: str) -> dict:
        """Apply selected healing strategy"""
        logger.info(f"Applying healing strategy: {strategy}")
        
        try:
            if strategy == "rotate_provider":
                return await self._rotate_provider()
            elif strategy == "reset_connection":
                return await self._reset_connections()
            elif strategy == "restart_component":
                return await self._restart_component()
            elif strategy == "fallback_mode":
                return await self._activate_fallback()
            else:  # retry_with_backoff
                return {"action": "wait", "duration": random.randint(5, 30)}
        except Exception as e:
            logger.error(f"Healing failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _rotate_provider(self) -> dict:
        """Rotate to fallback API provider"""
        logger.info("Rotating API provider")
        self.switch_to_fallback()
        return {"status": "success", "action": "provider_rotation"}
    
    async def _reset_connections(self) -> dict:
        """Reset all network connections"""
        logger.info("Resetting network connections")
        # Reset database connections
        from swarm.storage import MongoDBManager
        db = MongoDBManager()
        await db.reconnect()
        
        # Reset other connections
        return {"status": "success", "action": "connection_reset"}
    
    async def _restart_component(self) -> dict:
        """Restart a system component"""
        logger.warning("Restarting system component")
        # Select random component to restart (in real system would be more intelligent)
        components = ["scraper", "cleaner", "planner"]
        component = random.choice(components)
        
        # Simulate restart
        return {"status": "success", "action": "component_restart", "component": component}
    
    async def _activate_fallback(self) -> dict:
        """Activate fallback mode"""
        logger.critical("Activating fallback mode")
        # Implement fallback logic
        return {"status": "activated", "mode": "degraded_performance"}
    
    def _load_healing_strategies(self) -> dict:
        """Load healing strategies from config"""
        # In real implementation, load from YAML/JSON
        return {
            "rate_limit": "rotate_provider",
            "connection_error": "reset_connection",
            "timeout": "retry_with_backoff",
            "high_failure_rate": "restart_component",
            "critical_failure": "fallback_mode"
        }
