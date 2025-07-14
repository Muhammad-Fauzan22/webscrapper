import os, asyncio, aiohttp, json, random
from datetime import datetime
from pymongo import MongoClient

# === KONFIGURASI ===
MONGO_URI = os.getenv("MONGODB_URI")
SCRAPEOPS_KEY = os.getenv("SCRAPEOPS_API_KEY")
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36"
]

mongo = MongoClient(MONGO_URI)
db = mongo["scraper"]

# === FUNGSI SCRAPE ===
async def fetch(url):
    async with aiohttp.ClientSession() as s:
        headers = {"User-Agent": random.choice(UA_LIST)}
        async with s.get(url, headers=headers, timeout=15) as r:
            return await r.text()

async def run_cycle():
    targets = ["https://aseanenergy.org", "https://hydro.org"]
    for url in targets:
        html = await fetch(url)
        db.raw.insert_one({"url": url, "html": html, "ts": datetime.utcnow()})

if __name__ == "__main__":
    asyncio.run(run_cycle())
