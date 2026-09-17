"""arXiv API client with a simple on-disk response cache.

Search logic mirrors notebooks/01_explore_arxiv_api.ipynb.
"""

import hashlib
from pathlib import Path

import feedparser
import requests

from papermesh.cache import DATA_DIR, read_json, write_json

ARXIV_API_URL = "https://export.arxiv.org/api/query"
CACHE_DIR = DATA_DIR / "arxiv_cache"
REQUEST_TIMEOUT_SECONDS = 30


def _short_id(entry_id: str) -> str:
    """Turn 'http://arxiv.org/abs/2301.08243v2' into '2301.08243'."""
    short = entry_id.split("/abs/", 1)[-1]
    base, sep, version = short.rpartition("v")
    if sep and base and version.isdigit():
        return base
    return short


def _cache_path(query: str, max_results: int) -> Path:
    key = f"{query.strip().lower()}|{max_results}"
    query_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{query_hash}.json"


def parse_feed(feed_text: str) -> list[dict]:
    """Parse an arXiv Atom feed into paper dicts."""
    feed = feedparser.parse(feed_text)

    papers = []
    for entry in feed.entries:
        papers.append({
            "id": _short_id(entry.id),
            "title": " ".join(entry.title.split()),
            "abstract": " ".join(entry.summary.split()),
            "authors": [a.name for a in entry.authors],
            "published": entry.published,
            "link": entry.link,
        })
    return papers


def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """Search arXiv, reusing a cached response for the same query within 24h."""
    cache_path = _cache_path(query, max_results)
    cached = read_json(cache_path)
    if cached is not None:
        return cached

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    papers = parse_feed(response.text)
    write_json(cache_path, papers)
    return papers
