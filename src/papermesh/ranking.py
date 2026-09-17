"""Paper ranking: semantic relevance, optionally blended with citations and recency."""

import math
from datetime import date, datetime

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_RELEVANCE_WEIGHT = 0.6
DEFAULT_CITATION_WEIGHT = 0.25
DEFAULT_RECENCY_WEIGHT = 0.15
# A paper's recency score halves every this many years.
RECENCY_HALF_LIFE_YEARS = 3.0

_model = SentenceTransformer(MODEL_NAME)


def rank_papers(query: str, papers: list[dict]) -> list[dict]:
    """Return copies of `papers` with a relevance_score (0-1), sorted descending."""
    if not papers:
        return []

    query_emb = _model.encode([query], normalize_embeddings=True)
    abstract_embs = _model.encode(
        [p["abstract"] for p in papers], normalize_embeddings=True
    )
    # Normalized embeddings: dot product == cosine similarity. Clip negatives to 0.
    scores = np.clip(abstract_embs @ query_emb[0], 0.0, 1.0)

    ranked = [{**p, "relevance_score": float(s)} for p, s in zip(papers, scores)]
    ranked.sort(key=lambda p: p["relevance_score"], reverse=True)
    return ranked


def _citation_scores(papers: list[dict]) -> list[float]:
    """log(1 + citations), scaled so the most-cited paper in the set scores 1."""
    logs = [math.log1p(p.get("citation_count") or 0) for p in papers]
    top = max(logs, default=0.0)
    return [value / top if top > 0 else 0.0 for value in logs]


def _recency_score(published: str, today: date) -> float:
    published_on = datetime.fromisoformat(published).date()
    age_years = max((today - published_on).days, 0) / 365.25
    return 0.5 ** (age_years / RECENCY_HALF_LIFE_YEARS)


def weighted_rank(
    papers: list[dict],
    relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
    citation_weight: float = DEFAULT_CITATION_WEIGHT,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    today: date | None = None,
) -> list[dict]:
    """Return copies of `papers` with citation_score, recency_score, and a weighted score, sorted descending.

    Expects papers that already have relevance_score (from rank_papers). Weights are
    normalized to sum to 1; a missing citation_count counts as 0 citations.
    """
    today = today or date.today()
    total = relevance_weight + citation_weight + recency_weight
    if total <= 0:
        relevance_weight, citation_weight, recency_weight, total = 1.0, 0.0, 0.0, 1.0

    ranked = []
    for paper, citation_score in zip(papers, _citation_scores(papers)):
        recency_score = _recency_score(paper["published"], today)
        score = (
            relevance_weight * paper["relevance_score"]
            + citation_weight * citation_score
            + recency_weight * recency_score
        ) / total
        ranked.append({
            **paper,
            "citation_score": citation_score,
            "recency_score": recency_score,
            "score": score,
        })
    ranked.sort(key=lambda p: p["score"], reverse=True)
    return ranked
