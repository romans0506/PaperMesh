"""Semantic Scholar client: citation counts and reference lists for arXiv papers."""

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from papermesh.cache import DATA_DIR, read_json, write_json

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = ",".join([
    "citationCount",
    "year",
    "references.paperId",
    "references.title",
    "references.year",
    "references.citationCount",
    "references.externalIds",
])
CACHE_DIR = DATA_DIR / "s2_cache"
BATCH_SIZE = 500  # API maximum per batch request
REQUEST_TIMEOUT_SECONDS = 60
# Unauthenticated requests share a global rate limit, so 429s are routine; back off and retry.
RETRY_DELAYS_SECONDS = (1, 3, 8)

load_dotenv()


def _cache_path(arxiv_id: str) -> Path:
    return CACHE_DIR / f"{arxiv_id.replace('/', '_')}.json"


def _parse_paper(raw: dict | None) -> dict | None:
    if raw is None:
        return None
    references = []
    for ref in raw.get("references") or []:
        if not ref.get("paperId"):  # unresolved reference, nothing to link to
            continue
        references.append({
            "s2_id": ref["paperId"],
            "title": ref.get("title") or "",
            "year": ref.get("year"),
            "citation_count": ref.get("citationCount"),
            "arxiv_id": (ref.get("externalIds") or {}).get("ArXiv"),
        })
    return {
        "s2_id": raw["paperId"],
        "citation_count": raw.get("citationCount"),
        "year": raw.get("year"),
        "references": references,
    }


def _post_batch(ids: list[str]) -> list[dict | None]:
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        response = requests.post(
            BATCH_URL,
            params={"fields": FIELDS},
            json={"ids": ids},
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        retryable = response.status_code == 429 or response.status_code >= 500
        if not retryable or attempt == len(RETRY_DELAYS_SECONDS):
            break
        time.sleep(RETRY_DELAYS_SECONDS[attempt])
    response.raise_for_status()
    return response.json()


def fetch_citation_data(arxiv_ids: list[str]) -> dict[str, dict | None]:
    """Map each arXiv ID to its citation data, or None if Semantic Scholar doesn't know it.

    Results (including "not found") are cached per paper for 24h.
    Raises requests.RequestException if the API can't be reached.
    """
    result: dict[str, dict | None] = {}
    missing = []
    for arxiv_id in dict.fromkeys(arxiv_ids):
        cached = read_json(_cache_path(arxiv_id))
        if cached is not None:
            result[arxiv_id] = cached["paper"]
        else:
            missing.append(arxiv_id)

    for start in range(0, len(missing), BATCH_SIZE):
        chunk = missing[start:start + BATCH_SIZE]
        raw_papers = _post_batch([f"ARXIV:{arxiv_id}" for arxiv_id in chunk])
        for arxiv_id, raw in zip(chunk, raw_papers):
            paper = _parse_paper(raw)
            write_json(_cache_path(arxiv_id), {"paper": paper})
            result[arxiv_id] = paper
    return result


def add_citation_data(papers: list[dict], citation_data: dict[str, dict | None]) -> list[dict]:
    """Return copies of `papers` with citation_count (None if unknown), s2_id, and references."""
    enriched = []
    for paper in papers:
        data = citation_data.get(paper["id"]) or {}
        enriched.append({
            **paper,
            "s2_id": data.get("s2_id"),
            "citation_count": data.get("citation_count"),
            "references": data.get("references", []),
        })
    return enriched
