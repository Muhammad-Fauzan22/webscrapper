#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED DATA CLEANING AGENT
Version: 3.0.0
Created: 2025-07-17
Author: Data Processing Team
"""

import re
import logging
from .base_agent import BaseAgent

logger = logging.getLogger("DataCleaner")

class DataCleaner(BaseAgent):
    def __init__(self, primary_config, fallback_config, cost_optimizer):
        super().__init__(primary_config, fallback_config, cost_optimizer)
        self.min_quality = 0.7
    
    async def clean(self, raw_data: str) -> dict:
        """Clean and structure raw data"""
        # First try AI cleaning
        try:
            ai_result = await self._clean_with_ai(raw_data)
            
            if ai_result.get('quality_score', 0) >= self.min_quality:
                return ai_result
        except Exception as e:
            logger.warning(f"AI cleaning failed: {str(e)}")
        
        # Fallback to rule-based cleaning
        return self._basic_clean(raw_data)
    
    async def _clean_with_ai(self, raw_data: str) -> dict:
        """Use AI for advanced data cleaning"""
        prompt = self._build_cleaning_prompt(raw_data)
        
        response = await self.execute_with_fallback(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = response["choices"][0]["message"]["content"]
        result = self._parse_ai_response(content)
        result["tokens"] = response["usage"]["total_tokens"]
        
        return result
    
    def _build_cleaning_prompt(self, raw_data: str) -> str:
        """Construct prompt for AI cleaning"""
        return f"""
        Perform advanced cleaning and structuring on the following scraped content:
        
        Requirements:
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
        {raw_data[:10000]}{'... [truncated]' if len(raw_data) > 10000 else ''}
        """
    
    def _parse_ai_response(self, content: str) -> dict:
        """Parse AI response into structured data"""
        try:
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse AI response: {str(e)}")
            raise ValueError("Invalid AI response format")
    
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
