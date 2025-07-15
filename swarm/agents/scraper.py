#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED WEB SCRAPER AGENT
Version: 3.1.7
Created: 2025-07-15
Author: Data Acquisition Team
"""

import asyncio
import logging
import base64
import random
import time
from datetime import datetime
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

logger = logging.getLogger("ScraperAgent")

class Scraper:
    def __init__(self, timeout=30, max_retries=2):
        self.timeout = timeout * 1000  # Convert to ms
        self.max_retries = max_retries
        self.ua = UserAgent()
        
    async def execute(self, url: str, ai_config: dict) -> dict:
        """Execute scraping operation with retry logic"""
        result = {'status': 'PENDING', 'url': url}
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Scraping attempt {attempt} for {url}")
                scraped_data = await self._scrape_page(url)
                
                # Analyze content with AI
                if scraped_data['content']:
                    analysis = await self._analyze_content(
                        scraped_data['content'], 
                        ai_config
                    )
                    scraped_data['analysis'] = analysis
                
                result.update(scraped_data)
                result['status'] = 'SUCCESS'
                result['attempts'] = attempt
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {str(e)}")
                result['last_error'] = str(e)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        result['status'] = 'FAILURE'
        return result
        
    async def _scrape_page(self, url: str) -> dict:
        """Scrape web page using Playwright with advanced features"""
        async with async_playwright() as p:
            # Configure browser with stealth settings
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=self.ua.random,
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='Asia/Singapore',
                java_script_enabled=True,
                ignore_https_errors=True
            )
            
            # Configure stealth
            await context.add_init_script("""
                delete navigator.__proto__.webdriver;
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            """)
            
            page = await context.new_page()
            start_time = time.time()
            
            try:
                # Navigate with network monitoring
                await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
                
                # Wait for dynamic content
                await page.wait_for_load_state("networkidle", timeout=self.timeout)
                
                # Capture page content and screenshot
                content = await self._get_page_content(page)
                screenshot = await page.screenshot(type='png', full_page=True)
                
                # Extract metadata
                metadata = {
                    'title': await page.title(),
                    'url': page.url,
                    'load_time': time.time() - start_time,
                    'scrape_time': datetime.now().isoformat()
                }
                
                return {
                    'content': content,
                    'screenshot': base64.b64encode(screenshot).decode('utf-8'),
                    'metadata': metadata
                }
            finally:
                await browser.close()
    
    async def _get_page_content(self, page) -> str:
        """Extract and clean page content"""
        content = await page.content()
        
        # Remove unnecessary tags
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script, style, and other non-content tags
        for tag in soup(['script', 'style', 'noscript', 'meta', 'link']):
            tag.decompose()
            
        # Clean whitespace
        text = soup.get_text(separator='\n', strip=True)
        
        # Remove excessive newlines
        cleaned = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        return cleaned[:500000]  # Limit to 500KB
    
    async def _analyze_content(self, content: str, ai_config: dict) -> dict:
        """Analyze content with AI using structured output"""
        prompt = f"""
        Analyze the scraped web content and extract structured data:
        
        Content:
        {content[:10000]}{'... [truncated]' if len(content) > 10000 else ''}
        
        Required output format (JSON):
        {{
            "main_topic": "string (max 3 words)",
            "keywords": ["list", "of", "top", "5", "keywords"],
            "summary": "string (50-100 words)",
            "sentiment": "positive/neutral/negative",
            "language": "primary language code",
            "data_categories": ["list", "of", "relevant", "categories"]
        }}
        """
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": ai_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.2,
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
                timeout=30
            ) as response:
                data = await response.json()
                return json.loads(data["choices"][0]["message"]["content"])
