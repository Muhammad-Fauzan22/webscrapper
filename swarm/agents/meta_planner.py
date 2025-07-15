#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI META PLANNER AGENT
Version: 1.3.5
Created: 2025-07-15
Author: AI Strategy Team
"""

import aiohttp
import json
import logging
import re
from typing import List

logger = logging.getLogger("MetaPlanner")

class MetaPlanner:
    def __init__(self, max_targets=20, max_retries=3):
        self.max_targets = max_targets
        self.max_retries = max_retries
        
    async def generate_scrape_plan(self, topic: str, api_config: dict) -> List[str]:
        """Generate scraping plan using AI with fallback strategies"""
        for attempt in range(self.max_retries):
            try:
                return await self._generate_with_ai(topic, api_config)
            except Exception as e:
                logger.warning(f"Planning attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Fallback to predefined targets
        logger.error("AI planning failed, using fallback targets")
        return self._get_fallback_targets(topic)
    
    async def _generate_with_ai(self, topic: str, api_config: dict) -> List[str]:
        """Generate target list using AI service"""
        prompt = self._build_prompt(topic)
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": api_config["model"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert web scraping strategist specializing in ASEAN research data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
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
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error}")
                
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_response(content)
    
    def _build_prompt(self, topic: str) -> str:
        """Construct detailed prompt for target generation"""
        return f"""
        Generate a strategic web scraping plan for research on: {topic}
        
        Requirements:
        1. Focus on ASEAN region sources
        2. Prioritize authoritative sources in this order:
           - Government websites (.gov)
           - International organizations (.org)
           - Educational institutions (.edu)
           - Reputable news sources
        3. Include exactly {self.max_targets} distinct URLs
        4. Ensure URLs cover multiple ASEAN countries
        5. Include at least 3 data portal/repository sites
        6. Format output as a JSON array of strings
        
        Example output:
        [
            "https://data.aseanstats.org",
            "https://www.singapore.gov.sg/energy",
            "https://research.thailand.edu/renewables"
        ]
        """
    
    def _parse_response(self, content: str) -> List[str]:
        """Extract URLs from AI response with validation"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in response")
                
            urls = json.loads(json_match.group(0))
            
            # Validate URLs
            if not isinstance(urls, list):
                raise ValueError("Response is not a list")
                
            if len(urls) != self.max_targets:
                logger.warning(f"Expected {self.max_targets} URLs, got {len(urls)}")
                
            # Filter and validate URLs
            valid_urls = []
            for url in urls:
                if isinstance(url, str) and url.startswith(('http://', 'https://')):
                    valid_urls.append(url)
            
            if not valid_urls:
                raise ValueError("No valid URLs found in response")
                
            return valid_urls
        except Exception as e:
            logger.error(f"Response parsing failed: {str(e)}")
            raise
    
    def _get_fallback_targets(self, topic: str) -> List[str]:
        """Get predefined fallback targets based on topic"""
        base_targets = [
            "https://asean.org",
            "https://www.eria.org",
            "https://www.adb.org",
            "https://www.worldbank.org/en/region/eap",
            "https://data.worldbank.org"
        ]
        
        # Add topic-specific fallbacks
        if "energy" in topic.lower():
            base_targets.extend([
                "https://aseanenergy.org",
                "https://www.asean-renewables.org",
                "https://www.irena.org/asean",
                "https://www.energy.gov.sg",
                "https://www.doe.gov.ph"
            ])
        
        return base_targets[:self.max_targets]
