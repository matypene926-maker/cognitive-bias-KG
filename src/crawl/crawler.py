"""
src/crawl/crawler.py
====================
Web crawler for the Cognitive Bias Knowledge Graph project.
Collects pages from Wikipedia and educational sources on cognitive biases.

Pipeline:
  1. robots.txt verification (ethics)
  2. HTTP download (httpx)
  3. Main content extraction (trafilatura)
  4. Filter: pages < 500 words rejected
  5. Storage: JSONL {url, title, text, word_count, domain, timestamp}

Output: data/crawler_output.jsonl
"""

import httpx
import trafilatura
import json
import time
import re
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_WORDS       = 500
POLITENESS_DELAY = 1.5
REQUEST_TIMEOUT  = 15
OUTPUT_FILE      = "data/crawler_output.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (CognitiveBiasKG/1.0; university-project)",
    "Accept-Language": "en-US,en;q=0.9",
}

SEED_URLS = [
    "https://en.wikipedia.org/wiki/Cognitive_bias",
    "https://en.wikipedia.org/wiki/Confirmation_bias",
    "https://en.wikipedia.org/wiki/Echo_chamber_(media)",
    "https://en.wikipedia.org/wiki/Filter_bubble",
    "https://en.wikipedia.org/wiki/Dunning%E2%80%93Kruger_effect",
    "https://en.wikipedia.org/wiki/Availability_heuristic",
    "https://en.wikipedia.org/wiki/Anchoring_(cognitive_bias)",
    "https://en.wikipedia.org/wiki/Bandwagon_effect",
    "https://en.wikipedia.org/wiki/Framing_effect_(psychology)",
    "https://en.wikipedia.org/wiki/Misinformation",
    "https://en.wikipedia.org/wiki/Fake_news",
    "https://en.wikipedia.org/wiki/Algorithmic_radicalization",
    "https://en.wikipedia.org/wiki/Social_media_and_political_polarization",
    "https://en.wikipedia.org/wiki/Recommendation_system",
    "https://en.wikipedia.org/wiki/Political_polarization",
    "https://en.wikipedia.org/wiki/Disinformation",
    "https://en.wikipedia.org/wiki/In-group_favoritism",
    "https://en.wikipedia.org/wiki/Tribalism",
    "https://en.wikipedia.org/wiki/Motivated_reasoning",
    "https://en.wikipedia.org/wiki/Illusory_truth_effect",
]


def can_crawl(url: str) -> bool:
    """Check robots.txt — mandatory ethical practice."""
    try:
        parsed = urlparse(url)
        rp = RobotFileParser()
        rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch("*", url)
    except Exception:
        return True


def fetch_page(url: str, client: httpx.Client) -> dict | None:
    """Download and extract main content from a page."""
    if not can_crawl(url):
        logger.warning(f"Blocked by robots.txt: {url}")
        return None
    try:
        r    = client.get(url, timeout=REQUEST_TIMEOUT)
        html = r.text
        text = trafilatura.extract(
            html, include_tables=True,
            include_comments=False,
            favor_precision=True,
            no_fallback=False,
        )
        if not text or len(text.split()) < MIN_WORDS:
            return None
        m     = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        title = m.group(1).strip().replace("&amp;", "&") if m else url.split("/")[-1]
        return {
            "url":        url,
            "title":      title,
            "text":       text,
            "word_count": len(text.split()),
            "domain":     urlparse(url).netloc,
            "timestamp":  datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def run_crawler(
    seed_urls:   list[str] = SEED_URLS,
    output_file: str       = OUTPUT_FILE,
) -> list[dict]:
    """Main crawler pipeline."""
    Path("data").mkdir(exist_ok=True)
    Path(output_file).unlink(missing_ok=True)

    logger.info(f"Starting crawl — {len(seed_urls)} URLs")
    collected = []

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for i, url in enumerate(seed_urls, 1):
            label = url.split("/")[-1].replace("%E2%80%93", "-")[:45]
            logger.info(f"[{i:02d}/{len(seed_urls)}] {label}")
            doc = fetch_page(url, client)
            if doc:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                collected.append(doc)
                logger.info(f"  ✓ {doc['word_count']} words")
            else:
                logger.info("  ✗ skipped")
            if i < len(seed_urls):
                time.sleep(POLITENESS_DELAY)

    logger.info(f"Crawl done: {len(collected)}/{len(seed_urls)} pages collected")
    return collected


if __name__ == "__main__":
    docs = run_crawler()
    print(f"\n✓ {len(docs)} pages → {OUTPUT_FILE}")
