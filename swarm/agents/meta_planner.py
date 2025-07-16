#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
META PLANNER AGENT
Version: 3.0.0
Created: 2025-07-17
Author: Muhammad-Fauzan22
"""

import asyncio
import json
import logging
from typing import List, Dict
from .base_agent import BaseAgent

logger = logging.getLogger("MetaPlanner")

class MetaPlanner(BaseAgent):
    def __init__(self, primary_config, fallback_config, cost_optimizer):
        super().__init__(primary_config, fallback_config, cost_optimizer)
        self.max_targets = 20
        self.min_quality = 0.7
    
    async def generate_scrape_plan(self, topic: str) -> Dict:
        """Generate scraping plan with token optimization"""
        # Check cache first
        cache_key = f"plan_{topic.replace(' ', '_')}"
        cached = await self.cache_get(cache_key)
        if cached:
            logger.info("Using cached scraping plan")
            return cached
        
        # Prepare the optimized prompt
        prompt = self._build_optimized_prompt(topic)
        
        # Execute with token budget awareness
        response = await self.execute_with_fallback(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
            token_budget=self.token_budget_remaining
        )
        
        # Process response
        result = self._parse_response(response)
        result["tokens"] = response.get("usage", {}).get("total_tokens", 0)
        
        # Cache result for future use
        await self.cache_set(cache_key, result, ttl=86400)  # Cache for 24 hours
        
        return result
    
    def _build_optimized_prompt(self, topic: str) -> str:
        """Construct efficient prompt for target generation"""
        return f"""
        [SYSTEM]
        You are an expert web scraping strategist specializing in ASEAN research data.
        Generate a strategic scraping plan for: {topic}
        
        [REQUIREMENTS]
        1. Focus on ASEAN region sources
        2. Include exactly {self.max_targets} distinct URLs
        3. Prioritize authoritative sources (.gov, .edu, .org)
        4. Cover multiple ASEAN countries
        5. Include at least 3 data portals
        6. Format output as JSON: {{"targets": ["url1", "url2"]}}
        
        [RESPONSE FORMAT]
        JSON only, no additional text
        """
    
    def _parse_response(self, response: Dict) -> Dict:
        """Extract and validate plan from AI response"""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Clean JSON response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            plan = json.loads(json_str)
            
            # Validate structure
            if "targets" not in plan:
                raise ValueError("Invalid plan format: 'targets' key missing")
                
            if not isinstance(plan["targets"], list):
                raise ValueError("Invalid plan format: 'targets' should be a list")
                
            if len(plan["targets"]) < 5:  # Minimum 5 targets
                raise ValueError("Insufficient targets generated")
                
            return plan
        except Exception as e:
            logger.error(f"Plan parsing failed: {str(e)}")
            
            # Fallback to predefined targets
            return {
                "targets": self._get_fallback_targets(topic),
                "source": "fallback"
            }
    
    def _get_fallback_targets(self, topic: str) -> List[str]:
        """Get predefined fallback targets"""
        base_targets = [
            "https://asean.org",
            "https://www.eria.org",
            "https://www.adb.org",
            "https://data.worldbank.org/region/east-asia-and-pacific",
            "https://www.un.org/asia-pacific/"
        ]
        
        # Add topic-specific fallbacks
        topic_lower = topic.lower()
        if "energy" in topic_lower:
            base_targets.extend([
                "https://aseanenergy.org",
                "https://www.asean-renewables.org",
                "https://www.irena.org/asean",
                "https://www.energy.gov.sg",
                "https://www.doe.gov.ph"
            ])
        elif "economic" in topic_lower:
            base_targets.extend([
                "https://www.aseanstats.org",
                "https://www.imf.org/en/Countries/ResRep/asean",
                "https://www.worldbank.org/en/region/eap",
                "https://www.mof.gov.sg",
                "https://www.bsp.gov.ph"
            ])
        
        return base_targets[:self.max_targets]
