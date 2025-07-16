#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED WEB SCRAPER AGENT
Version: 3.0.0
Created: 2025-07-17
Author: Scraping Team
"""

import asyncio
import random
import logging
from playwright.async_api import async_playwright
from .base_agent import BaseAgent
from swarm.utils import RetryHandler
from swarm.storage import HFCacheManager

logger = logging.getLogger("Scraper")

class Scraper(BaseAgent):
    def __init__(self, primary_config, fallback_config, cost_optimizer):
        super().__init__(primary_config, fallback_config, cost_optimizer)
        self.cache = HFCacheManager()
        self.user_agents = self._load_user_agents()
        self.proxies = self._load_proxies()
        self.headless = True  # Run in headless mode for cloud deployment
    
    async def scrape_url(self, url: str) -> dict:
        """Scrape content from a URL with advanced techniques"""
        # Check cache first
        cache_key = f"scrape_{hash(url)}"
        cached = self.cache.get(cache_key, max_age_hours=24)
        if cached:
            logger.info(f"Using cached content for {url}")
            return cached
        
        # Prepare result object
        result = {
            "url": url,
            "content": "",
            "status": "pending",
            "tokens": 0
        }
        
        # Select random user agent and proxy
        user_agent = random.choice(self.user_agents)
        proxy = random.choice(self.proxies) if self.proxies else None
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                proxy=proxy
            )
            
            context = await browser.new_context(user_agent=user_agent)
            page = await context.new_page()
            
            try:
                # Navigate to page
                await page.goto(url, timeout=60000)
                await page.wait_for_load_state("networkidle", timeout=30000)
                
                # Handle consent dialogs
                await self._handle_consent_dialogs(page)
                
                # Extract main content
                content = await self._extract_main_content(page)
                
                # Handle pagination
                paginated_content = await self._handle_pagination(page)
                full_content = content + "\n".join(paginated_content)
                
                # Update result
                result["content"] = full_content
                result["status"] = "success"
                result["tokens"] = len(full_content) // 4  # Estimate tokens
                
                # Cache result
                self.cache.set(cache_key, result)
                
                return result
            except Exception as e:
                logger.error(f"Scraping failed for {url}: {str(e)}")
                result["status"] = "error"
                result["error"] = str(e)
                return result
            finally:
                await browser.close()
    
    async def _extract_main_content(self, page) -> str:
        """Extract main content using AI-enhanced detection"""
        # First try: Use AI to find main content
        try:
            ai_content = await self._ai_extract_content(page)
            if ai_content:
                return ai_content
        except:
            pass
        
        # Fallback: Use common content selectors
        content_selectors = [
            "main", "article", ".main-content", ".content", "#content"
        ]
        
        for selector in content_selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                content = ""
                for element in elements:
                    content += await element.inner_text() + "\n\n"
                return content
        
        # Final fallback: Get all text
        return await page.content()
    
    async def _ai_extract_content(self, page) -> str:
        """Use AI to extract main content"""
        # Get page structure
        structure = await self._get_page_structure(page)
        
        # Ask AI to identify main content
        prompt = f"""
        Given this page structure, identify the main content container:
        
        {structure}
        
        Respond only with the most appropriate CSS selector for the main content.
        """
        
        response = await self.execute_with_fallback(
            prompt=prompt,
            max_tokens=50,
            temperature=0.1
        )
        
        selector = response["choices"][0]["message"]["content"].strip()
        
        # Extract content using AI-selected selector
        if selector:
            element = await page.query_selector(selector)
            if element:
                return await element.inner_text()
        
        return None
    
    async def _get_page_structure(self, page) -> str:
        """Get simplified page structure for AI analysis"""
        elements = await page.query_selector_all("body *")
        structure = []
        
        for element in elements:
            tag = await element.evaluate("el => el.tagName.toLowerCase()")
            classes = await element.get_attribute("class") or ""
            id_attr = await element.get_attribute("id") or ""
            
            if classes or id_attr:
                selector = f"{tag}"
                if id_attr:
                    selector += f"#{id_attr}"
                if classes:
                    selector += f".{'.'.join(classes.split()[:2])}"  # Limit classes
                
                structure.append(selector)
        
        return "\n".join(set(structure))  # Deduplicate
    
    async def _handle_consent_dialogs(self, page):
        """Automatically handle common consent dialogs"""
        # Try to accept cookies
        accept_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            'button:has-text("OK")',
            'button:has-text("I Accept")',
            'button#accept-cookies'
        ]
        
        for selector in accept_selectors:
            if await page.query_selector(selector):
                await page.click(selector, timeout=5000)
                await asyncio.sleep(1)  # Wait for dialog to close
                break
    
    async def _handle_pagination(self, page) -> list:
        """Automatically navigate through pagination"""
        contents = []
        page_count = 0
        max_pages = 3
        
        while page_count < max_pages:
            next_button = await page.query_selector("a.next, a[rel='next']")
            if not next_button:
                break
                
            await next_button.click()
            await page.wait_for_load_state("networkidle", timeout=20000)
            contents.append(await page.content())
            page_count += 1
            
        return contents
    
    def _load_user_agents(self) -> list:
        """Load user agents from file or default list"""
        # In real implementation, load from file
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
        ]
    
    def _load_proxies(self) -> list:
        """Load proxies from environment or service"""
        proxy_str = os.getenv("PROXY_LIST", "")
        if proxy_str:
            return proxy_str.split(",")
        return None
