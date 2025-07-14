import asyncio
import logging
from datetime import datetime
from playwright.async_api import async_playwright
import aiohttp

logger = logging.getLogger("Scraper")

class Scraper:
    @staticmethod
    async def execute(session: aiohttp.ClientSession, url: str, ai_config: dict) -> dict:
        """Scrape and analyze content with DeepSeek AI"""
        try:
            logger.info(f"Scraping: {url}")
            
            # 1. Render page with Playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=15000)
                    content = await page.content()
                    
                    # Capture screenshot for debugging
                    screenshot = await page.screenshot(type='png')
                except Exception as e:
                    logger.warning(f"Page load failed: {str(e)}")
                    raise e
                finally:
                    await browser.close()
                
            # 2. Analyze with DeepSeek
            analysis = await Scraper.analyze_content(content, ai_config)
            
            return {
                "url": url,
                "content": content,
                "screenshot": screenshot,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            return e
            
    @staticmethod
    async def analyze_content(content: str, ai_config: dict) -> dict:
        """Analyze scraped content with DeepSeek AI"""
        prompt = f"""
        Analisis konten halaman web berikut dan ekstrak:
        1. Topik utama (max 3 kata)
        2. 5 kata kunci terpenting
        3. Ringkasan 50 kata
        4. Sentimen (positif/netral/negatif)
        5. Bahasa utama
        
        Format output: JSON
        
        Konten:
        {content[:10000]}... [truncated]
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": ai_config["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.2
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
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Analysis API error: {error}")
                        return {}
                        
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {}
