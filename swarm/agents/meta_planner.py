import aiohttp
import json
import logging

logger = logging.getLogger("MetaPlanner")

class MetaPlanner:
    @staticmethod
    async def generate_scrape_plan(topic: str, api_config: dict) -> list:
        """Generate scraping plan using Cypher Alpha"""
        prompt = f"""
        Sebagai AI perencana web scraping ahli, buatkan rencana scraping untuk topik: 
        '{topic}'. Rencana harus mencakup 20 website terpercaya di ASEAN, 
        dengan prioritas situs pemerintah (.gov), organisasi internasional (.org), 
        dan universitas (.edu). Format output: JSON list of URLs.
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": api_config["model"],
                    "messages": [{"role": "user", "content": prompt}],
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
                        logger.error(f"API error: {error}")
                        return []
                        
                    data = await response.json()
                    return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"Planning failed: {str(e)}")
            return []
