#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED COST OPTIMIZATION ENGINE
Version: 3.0.0
Created: 2025-07-17
Author: Financial Efficiency Team
"""

import os
import yaml
import logging
from datetime import datetime, timedelta
from swarm.utils import Logger

logger = Logger(name="CostOptimizer")

class CostOptimizer:
    def __init__(self):
        self.config_path = "configs/cost_config.yaml"
        self.cost_data = {}
        self.daily_budget = float(os.getenv("DAILY_BUDGET", 0.1))  # $0.1/day default
        self._load_config()
    
    def _load_config(self):
        """Load cost optimization configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info("Cost optimization config loaded")
            else:
                self.config = {
                    "providers": {
                        "primary": {"cost_per_token": 0.00002, "daily_limit": 5000},
                        "fallback": {"cost_per_token": 0.00001, "daily_limit": 10000}
                    },
                    "auto_switch_threshold": 0.8  # Switch at 80% of budget
                }
        except Exception as e:
            logger.error(f"Config load failed: {str(e)}")
            self.config = {}
    
    def track_usage(self, provider: str, tokens: int):
        """Track token usage and calculate cost"""
        if tokens <= 0:
            return
        
        # Get provider cost config
        provider_config = self.config["providers"].get(provider, {})
        cost_per_token = provider_config.get("cost_per_token", 0.00002)
        cost = tokens * cost_per_token
        
        # Update daily tracking
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.cost_data:
            self.cost_data[today] = {"total_cost": 0.0, "providers": {}}
        
        if provider not in self.cost_data[today]["providers"]:
            self.cost_data[today]["providers"][provider] = 0.0
        
        self.cost_data[today]["providers"][provider] += cost
        self.cost_data[today]["total_cost"] += cost
        
        logger.info(f"Tracked usage: {tokens} tokens on {provider} = ${cost:.6f}")
    
    def get_daily_cost(self) -> float:
        """Get total cost for current day"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.cost_data.get(today, {}).get("total_cost", 0.0)
    
    def get_remaining_budget(self) -> float:
        """Get remaining daily budget"""
        return self.daily_budget - self.get_daily_cost()
    
    def should_switch(self, provider: str) -> bool:
        """Determine if should switch from current provider"""
        today = datetime.now().strftime("%Y-%m-%d")
        provider_cost = self.cost_data.get(today, {}).get("providers", {}).get(provider, 0.0)
        
        # Check if exceeded provider-specific limit
        provider_limit = self.config["providers"].get(provider, {}).get("daily_limit", float('inf'))
        if provider_cost > provider_limit:
            return True
        
        # Check if approaching total budget
        total_cost = self.get_daily_cost()
        if total_cost > self.daily_budget * self.config.get("auto_switch_threshold", 0.8):
            return True
        
        return False
    
    def get_cost_report(self, period: str = "daily") -> dict:
        """Generate cost report"""
        report = {}
        
        if period == "daily":
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            report = {
                "today": self.cost_data.get(today, {}),
                "yesterday": self.cost_data.get(yesterday, {}),
                "budget": self.daily_budget,
                "remaining": self.daily_budget - self.get_daily_cost()
            }
        elif period == "weekly":
            # Implement weekly aggregation
            pass
        
        return report
    
    def optimize_api_calls(self, plan: dict) -> dict:
        """Optimize API call plan based on cost"""
        optimized_plan = {}
        provider_limits = self.config["providers"]
        
        for service, requests in plan.items():
            provider = requests["provider"]
            max_requests = provider_limits[provider].get("daily_limit", 1000)
            
            # Apply limit
            optimized_requests = requests["count"] if requests["count"] < max_requests else max_requests
            optimized_plan[service] = {
                "provider": provider,
                "optimized_count": optimized_requests,
                "cost_savings": (requests["count"] - optimized_requests) * 
                                provider_limits[provider]["cost_per_token"]
            }
        
        return optimized_plan
