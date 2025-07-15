#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SELF-HEALING AGENT
Version: 2.0.8
Created: 2025-07-15
Author: System Resilience Team
"""

import aiohttp
import json
import logging
import re
import os

logger = logging.getLogger("HealerAgent")

class Healer:
    def __init__(self, config_path="healing_strategies.json"):
        self.strategies = self._load_strategies(config_path)
        
    async def diagnose(self, failures: list, claude_config: dict, cypher_config: dict) -> dict:
        """Diagnose system failures and generate solutions"""
        # Step 1: Classify failures
        failure_summary = await self._classify_failures(failures, claude_config)
        
        # Step 2: Generate solutions
        solutions = await self._generate_solutions(failure_summary, cypher_config)
        
        # Step 3: Match with known strategies
        matched_solutions = []
        for solution in solutions:
            matched = self._match_known_strategy(solution)
            if matched:
                matched_solutions.append(matched)
            else:
                matched_solutions.append({
                    "type": "NEW_SOLUTION",
                    "description": solution,
                    "confidence": 0.7
                })
                
        return {
            "failure_summary": failure_summary,
            "solutions": matched_solutions
        }
    
    async def apply_solutions(self, solutions: list):
        """Apply healing solutions to the system"""
        for solution in solutions:
            try:
                if solution['type'] == "CONFIG_UPDATE":
                    self._apply_config_update(solution['details'])
                elif solution['type'] == "CODE_PATCH":
                    self._apply_code_patch(solution['details'])
                elif solution['type'] == "RESOURCE_ADJUSTMENT":
                    self._apply_resource_adjustment(solution['details'])
                elif solution['type'] == "RETRY_STRATEGY":
                    self._apply_retry_strategy(solution['details'])
                elif solution['type'] == "FALLBACK_ACTIVATION":
                    self._activate_fallback(solution['details'])
                    
                logger.info(f"Applied solution: {solution['type']}")
            except Exception as e:
                logger.error(f"Solution application failed: {str(e)}")
    
    async def _classify_failures(self, failures: list, api_config: dict) -> str:
        """Classify failures using AI"""
        failure_samples = "\n".join([str(f)[:500] for f in failures[:5]])
        
        prompt = f"""
        Classify these system failures and provide a diagnostic summary:
        
        Failures:
        {failure_samples}
        
        Classification instructions:
        1. Identify failure patterns
        2. Categorize by system component
        3. Determine root cause likelihood
        4. Assess severity (1-5)
        5. Output format:
            {{
                "patterns": ["list", "of", "patterns"],
                "categories": ["component", "names"],
                "root_cause_analysis": "text",
                "severity": integer
            }}
        """
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": api_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {api_config['key']}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                api_config["endpoint"],
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    
    async def _generate_solutions(self, diagnosis: str, api_config: dict) -> list:
        """Generate technical solutions using AI"""
        prompt = f"""
        Based on this system diagnosis, provide technical solutions:
        
        {diagnosis}
        
        Solution requirements:
        1. Provide 3-5 specific solutions
        2. Each solution should include:
           - Implementation steps
           - Expected impact
           - Risk assessment
        3. Prioritize solutions by effectiveness
        4. Output format: JSON array of solution objects
        """
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": api_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {api_config['key']}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                api_config["endpoint"],
                json=payload,
                headers=headers,
                timeout=45
            ) as response:
                data = await response.json()
                return json.loads(data["choices"][0]["message"]["content"])["solutions"]
    
    def _load_strategies(self, path: str) -> dict:
        """Load known healing strategies from file"""
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def _match_known_strategy(self, solution: str) -> dict:
        """Match solution to known healing strategies"""
        # Simplified matching logic - in practice would use NLP matching
        solution_lower = solution.lower()
        
        if "timeout" in solution_lower:
            return self.strategies.get("timeout_adjustment", {
                "type": "CONFIG_UPDATE",
                "description": "Increase timeout thresholds",
                "parameters": {"timeout": "increase"},
                "confidence": 0.9
            })
        elif "memory" in solution_lower:
            return self.strategies.get("memory_optimization", {
                "type": "RESOURCE_ADJUSTMENT",
                "description": "Optimize memory usage",
                "parameters": {"memory_limit": "increase"},
                "confidence": 0.85
            })
        elif "retry" in solution_lower:
            return self.strategies.get("retry_strategy", {
                "type": "RETRY_STRATEGY",
                "description": "Implement exponential backoff",
                "parameters": {"max_retries": 3, "backoff_factor": 2},
                "confidence": 0.8
            })
        
        return None
    
    def _apply_config_update(self, details: dict):
        """Apply configuration update strategy"""
        # Implementation would modify configuration files
        logger.info(f"Applying config update: {details}")
    
    def _apply_code_patch(self, details: dict):
        """Apply code patch strategy"""
        # Implementation would modify source code
        logger.info(f"Applying code patch: {details}")
    
    def _apply_resource_adjustment(self, details: dict):
        """Apply resource adjustment strategy"""
        # Implementation would adjust resource allocation
        logger.info(f"Applying resource adjustment: {details}")
    
    def _apply_retry_strategy(self, details: dict):
        """Apply retry strategy"""
        # Implementation would update retry logic
        logger.info(f"Applying retry strategy: {details}")
    
    def _activate_fallback(self, details: dict):
        """Activate fallback mechanism"""
        # Implementation would enable fallback systems
        logger.info(f"Activating fallback: {details}")
