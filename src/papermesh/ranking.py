"""Relevance-only ranking via sentence-transformer embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

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
