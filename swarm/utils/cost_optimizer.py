#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COST OPTIMIZATION ENGINE
Version: 2.0.0
Created: 2025-07-17
Author: Financial Efficiency Team
"""

import os
import logging
import yaml
from datetime import datetime, timedelta
from .logger import Logger

logger = Logger(name="CostOptimizer")

class CostOptimizer:
    def __init__(self):
        self.config_path = "configs/api_priority.yaml"
        self.cost_data = {}
        self.daily_budget = float(os.getenv("DAILY_BUDGET", 0.1))  # $0.1/day default
        self._load_config()
    
    def _load_config(self):
        """Load API priority configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info("API priority config loaded")
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            self.config = {
                "priority_chain": [
                    {"name": "cypher", "cost_per_token": 0.00002},
                    {"name": "deepseek", "cost_per_token": 0.000015},
                    {"name": "claude", "cost_per_token": 0.000025},
                    {"name": "huggingface", "cost_per_token": 0.0}
                ]
            }
    
    def track_usage(self, provider: str, tokens: int):
        """Track token usage and calculate cost"""
        if not tokens:
            return
        
        # Find provider cost
        cost_per_token = next(
            (p["cost_per_token"] for p in self.config["priority_chain"] 
            if p["name"] == provider
        ), 0.00002)  # Default if not found
        
        cost = tokens * cost_per_token
        
        # Update daily tracking
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.cost_data:
            self.cost_data[today] = {}
        
        if provider not in self.cost_data[today]:
            self.cost_data[today][provider] = 0.0
        
        self.cost_data[today][provider] += cost
        
        logger.info(
            f"Tracked usage: {tokens} tokens on {provider} = ${cost:.6f}"
        )
        
        # Check if we need to switch providers
        if self.cost_data[today][provider] > self.daily_budget / 3:
            self._activate_fallback(provider)
    
    def _activate_fallback(self, provider: str):
        """Activate fallback for a provider"""
        provider_config = next(
            p for p in self.config["priority_chain"] 
            if p["name"] == provider
        )
        
        if "fallback" not in provider_config:
            logger.warning(f"No fallback defined for {provider}")
            return
        
        fallback = provider_config["fallback"]
        logger.info(
            f"Cost threshold exceeded for {provider}. Switching to {fallback}"
        )
        
        # Update config priority
        self.config["priority_chain"] = [
            p for p in self.config["priority_chain"] 
            if p["name"] != provider
        ]
        
        # Add fallback to the end of the chain
        self.config["priority_chain"].append(
            next(
                p for p in self.config["priority_chain"] 
                if p["name"] == fallback
            )
        )
        
        # Save updated config
        self._save_config()
    
    def _save_config(self):
        """Save updated configuration"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f)
            logger.info("Updated API priority config saved")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
    
    def get_current_provider(self, service_type: str) -> str:
        """Get current preferred provider for a service type"""
        # Simplified logic - in real implementation would map service types
        return self.config["priority_chain"][0]["name"]
    
    def get_daily_cost(self) -> float:
        """Get total cost for current day"""
        today = datetime.now().strftime("%Y-%m-%d")
        return sum(self.cost_data.get(today, {}).values())
    
    def get_provider_cost(self, provider: str) -> float:
        """Get cost for a specific provider today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.cost_data.get(today, {}).get(provider, 0.0)
    
    def get_cost_report(self) -> dict:
        """Generate comprehensive cost report"""
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        return {
            "today": self.cost_data.get(today, {}),
            "yesterday": self.cost_data.get(yesterday, {}),
            "daily_budget": self.daily_budget,
            "remaining_budget": self.daily_budget - self.get_daily_cost()
        }
