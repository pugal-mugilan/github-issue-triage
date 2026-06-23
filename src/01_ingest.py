"""
01_ingest.py — Fetch closed issues from a set of GitHub repos.

Saves raw JSON per repo to data/raw/{owner__repo}.json with full API
fidelity — no filtering, no transformations. Cleaning happens in 02_clean.py.

Why the strict separation: scraping is slow and irreplaceable (live data
changes daily). If a cleaning bug surfaces later, we should be able to fix
it without re-hitting the API. So this script only captures and saves.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------- Config ----------

REPOS = [
    ("huggingface",  "transformers"),    # train
    ("pytorch",      "pytorch"),         # train
    ("langchain-ai", "langchain"),       # train
    ("fastapi",      "fastapi"),         # held-out OOD
    ("scikit-learn", "scikit-learn"),    # held-out OOD
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SINCE_DAYS = 540                          # ~18 months of history per repo
SINCE_DATE = (datetime.now(timezone.utc) - timedelta(days=SINCE_DAYS)).isoformat()

PER_PAGE = 100
GITHUB_API = "https://api.github.com"

# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("ingest")

# ---------- Auth ----------

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN not set. Generate one at github.com/settings/tokens "
        "(no scopes needed for public repos), then add to .env."
    )

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ---------- HTTP helper with retry + rate-limit handling ----------

def fetch_page(url, params=None):
    """
    Fetch one page. Returns (json_body, next_url_or_None).
    Retries on 5xx and rate-limit (403/429) with exponential backoff.
    """
    for attempt in range(5):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if resp.status_code == 200:
            # The 'requests' library parses the Link header for us into resp.links.
            next_url = resp.links.get("next", {}).get("url")

            # Be polite: if we're running low on rate-limit budget, sleep until reset.
            remaining = int(resp.headers.get("X-RateLimit-Remaining", "9999"))
            if remaining < 50:
                reset_at = int(resp.headers.get("X-RateLimit-Reset", "0"))
                wait_s = max(0, reset_at - int(time.time())) + 1
                log.warning(f"Rate limit low ({remaining}). Sleeping {wait_s}s.")
                time.sleep(wait_s)

            return resp.json(), next_url

        if resp.status_code in (403, 429):
            # Rate-limited. Wait until reset.
            reset_at = int(resp.headers.get("X-RateLimit-Reset", "0"))
            wait_s = max(1, reset_at - int(time.time())) + 1
            log.warning(f"Got {resp.status_code}. Sleeping {wait_s}s for reset.")
            time.sleep(wait_s)
            continue

        if 500 <= resp.status_code < 600:
            # Server error. Exponential backoff.
            wait_s = 2 ** attempt
            log.warning(f"Server returned {resp.status_code}. Retry in {wait_s}s.")
            time.sleep(wait_s)
            continue

        # Anything else (404, 401, etc.): fail loudly. Don't retry.
        resp.raise_for_status()

    raise RuntimeError(f"Failed to fetch {url} after 5 attempts")


def fetch_all_pages(initial_url, params):
    """Walk Link 'next' until exhausted, returning a flat list of items."""
    results = []
    url = initial_url
    current_params = params
    page_num = 0

    while url:
        page_num += 1
        body, next_url = fetch_page(url, params=current_params)
        results.extend(body)
        log.info(f"  page {page_num}: +{len(body)} items (running total {len(results)})")
        url = next_url
        current_params = None  # next_url already contains the params encoded
    return results


# ---------- Per-repo scrape ----------

def scrape_repo(owner, repo):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
    params = {
        "state": "closed",
        "per_page": PER_PAGE,
        "since": SINCE_DATE,         # filters by updated_at, NOT created_at — final filter in 02_clean.py
        "sort": "created",
        "direction": "desc",
    }
    log.info(f"Scraping {owner}/{repo} since {SINCE_DATE}")
    t0 = time.time()
    items = fetch_all_pages(url, params)
    elapsed = time.time() - t0
    log.info(f"  done: {len(items)} items in {elapsed:.1f}s")
    return items


# ---------- Main ----------

def main():
    metadata = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "since_date": SINCE_DATE,
        "per_repo_counts": {},
    }
    total = 0

    for owner, repo in REPOS:
        items = scrape_repo(owner, repo)
        out_path = OUTPUT_DIR / f"{owner}__{repo}.json"
        with open(out_path, "w") as f:
            json.dump(items, f)
        size_mb = out_path.stat().st_size / 1e6
        log.info(f"  saved {out_path}  ({size_mb:.1f} MB)")
        metadata["per_repo_counts"][f"{owner}/{repo}"] = len(items)
        total += len(items)

    with open(OUTPUT_DIR / "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Total items across {len(REPOS)} repos: {total}")


if __name__ == "__main__":
    main()