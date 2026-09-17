from datetime import date

import pytest

from papermesh.ranking import rank_papers, weighted_rank


def test_rank_papers_orders_by_relevance() -> None:
    papers = [
        {
            "id": "low",
            "title": "Sourdough baking",
            "abstract": "A guide to baking sourdough bread with a wild yeast starter and long fermentation.",
        },
        {
            "id": "high",
            "title": "Graph neural networks for molecules",
            "abstract": "We propose a graph neural network that predicts molecular properties "
            "by passing messages between atoms in a molecular graph.",
        },
        {
            "id": "mid",
            "title": "Convolutional networks for images",
            "abstract": "We train a deep convolutional neural network for image classification.",
        },
    ]

    ranked = rank_papers("graph neural networks for molecular property prediction", papers)

    assert [p["id"] for p in ranked] == ["high", "mid", "low"]
    assert all(0.0 <= p["relevance_score"] <= 1.0 for p in ranked)
    assert ranked[0]["relevance_score"] > ranked[-1]["relevance_score"]


def _scored(paper_id: str, relevance: float, citations: int | None, published: str) -> dict:
    return {"id": paper_id, "relevance_score": relevance, "citation_count": citations, "published": published}


def test_weighted_rank_blends_relevance_citations_and_recency() -> None:
    today = date(2026, 1, 1)
    papers = [
        _scored("relevant", 0.9, 0, "2015-01-01T00:00:00Z"),
        _scored("famous", 0.5, 10000, "2015-01-01T00:00:00Z"),
        _scored("fresh", 0.5, None, "2025-12-01T00:00:00Z"),
    ]

    by_relevance = weighted_rank(papers, 1.0, 0.0, 0.0, today=today)
    by_citations = weighted_rank(papers, 0.0, 1.0, 0.0, today=today)
    by_recency = weighted_rank(papers, 0.0, 0.0, 1.0, today=today)

    assert by_relevance[0]["id"] == "relevant"
    assert by_citations[0]["id"] == "famous"
    assert by_recency[0]["id"] == "fresh"


def test_weighted_rank_scores() -> None:
    today = date(2026, 1, 1)
    papers = [
        _scored("a", 0.8, 99, "2023-01-01T00:00:00Z"),
        _scored("b", 0.4, 9, "2026-01-01T00:00:00Z"),
    ]

    ranked = {p["id"]: p for p in weighted_rank(papers, 2.0, 1.0, 1.0, today=today)}

    assert ranked["a"]["citation_score"] == pytest.approx(1.0)
    assert ranked["b"]["citation_score"] == pytest.approx(0.5)  # log(10) / log(100)
    assert ranked["a"]["recency_score"] == pytest.approx(0.5, abs=0.01)  # ~3 years = one half-life
    assert ranked["b"]["recency_score"] == pytest.approx(1.0)
    # Weights 2:1:1 normalize to 0.5 / 0.25 / 0.25.
    assert ranked["b"]["score"] == pytest.approx(0.5 * 0.4 + 0.25 * 0.5 + 0.25 * 1.0)
