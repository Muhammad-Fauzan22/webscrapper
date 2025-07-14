import os, asyncio, aiohttp, random
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account

# === KONFIGURASI ===
DRIVE_FOLDER = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
SCRAPEOPS_KEY = os.getenv("SCRAPEOPS_API_KEY")
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36"
]

# Google Drive Setup
def build_drive_service():
    credentials = service_account.Credentials.from_service_account_file(
        'service_account.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=credentials)

drive_service = build_drive_service()

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
        file_metadata = {
            'name': f'scrape_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.html',
            'parents': [DRIVE_FOLDER]
        }
        media = MediaFileUpload.fromString(html, mimetype='text/html')
        drive_service.files().create(body=file_metadata, media_body=media).execute()

if __name__ == "__main__":
    asyncio.run(run_cycle())
