#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-POWERED DATA CLEANER
Version: 1.2.3
Created: 2025-07-15
Author: Data Processing Team
"""

import aiohttp
import json
import logging
import re

logger = logging.getLogger("DataCleaner")

class DataCleaner:
    def __init__(self, min_quality=0.7):
        self.min_quality = min_quality
        
    async def clean(self, raw_data: str, ai_config: dict) -> dict:
        """Clean and structure raw data using AI with fallback"""
        try:
            return await self._clean_with_ai(raw_data, ai_config)
        except Exception as e:
            logger.error(f"AI cleaning failed: {str(e)}, using fallback")
            return self._basic_clean(raw_data)
    
    async def _clean_with_ai(self, raw_data: str, ai_config: dict) -> dict:
        """Use AI for advanced data cleaning and structuring"""
        prompt = f"""
        Perform advanced cleaning and structuring on the following scraped content:
        
        1. Remove any irrelevant content (ads, navigation, etc.)
        2. Extract core informational content
        3. Structure into logical sections
        4. Identify key facts and statistics
        5. Preserve all numerical data and tables
        6. Assess data quality (0.0-1.0)
        7. Output in JSON format:
            {{
                "content": "cleaned text",
                "structure": {{
                    "sections": [
                        {{"title": "section title", "content": "section content"}}
                    ]
                }},
                "key_facts": ["list", "of", "important", "facts"],
                "statistics": [{{"value": number, "unit": "string", "description": "text"}}],
                "quality_score": float
            }}
        
        Content:
        {raw_data[:15000]}{'... [truncated]' if len(raw_data) > 15000 else ''}
        """
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": ai_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {ai_config['key']}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                ai_config["endpoint"],
                json=payload,
                headers=headers,
                timeout=60
            ) as response:
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                
                # Validate quality
                if result.get('quality_score', 0) < self.min_quality:
                    raise ValueError(f"Low quality score: {result['quality_score']}")
                
                return result
    
    def _basic_clean(self, raw_data: str) -> dict:
        """Basic cleaning fallback without AI"""
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', raw_data)
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Simple sentence segmentation
        sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean_text)]
        
        return {
            "content": clean_text,
            "quality_score": 0.5,
            "key_facts": sentences[:5],
            "statistics": self._extract_basic_stats(clean_text)
        }
    
    def _extract_basic_stats(self, text: str) -> list:
        """Extract basic statistics from text"""
        stats = []
        
        # Find percentages
        for match in re.finditer(r'(\d+(?:\.\d+)?%)', text):
            stats.append({
                "value": match.group(1),
                "unit": "percent",
                "description": "Found in text"
            })
            
        # Find numbers with units
        for match in re.finditer(r'(\d+(?:,\d+)*(?:\.\d+)?)\s?(MW|GW|kg|tons?|USD|\$)', text):
            stats.append({
                "value": match.group(1).replace(',', ''),
                "unit": match.group(2),
                "description": "Found in text"
            })
            
        return stats
