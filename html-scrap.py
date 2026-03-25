import httpx
import os

SECTION_URL = "https://www.ecfr.gov/current/title-34/subtitle-B/chapter-VI/part-668/section-668.32"

SAVE_DIR = "data/raw_html/ecfr"
os.makedirs(SAVE_DIR, exist_ok=True)
RAW_HTML_PATH = f"{SAVE_DIR}/section_668_32.html"


async def fetch_page(url):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        res = await client.get(url)
        res.raise_for_status()
        return res.text


async def fetch_ecfr_html():
    html = await fetch_page(SECTION_URL)
    with open(RAW_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved raw HTML to: {RAW_HTML_PATH}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_ecfr_html())