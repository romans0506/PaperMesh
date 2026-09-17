from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from papermesh import semantic_scholar
from papermesh.semantic_scholar import add_citation_data, fetch_citation_data

SAMPLE_BATCH = [
    {
        "paperId": "s2-jepa",
        "citationCount": 1200,
        "year": 2023,
        "references": [
            {
                "paperId": "s2-vit",
                "title": "An Image is Worth 16x16 Words",
                "year": 2020,
                "citationCount": 50000,
                "externalIds": {"ArXiv": "2010.11929", "DBLP": "x"},
            },
            {"paperId": None, "title": "Unresolved reference", "year": None, "citationCount": None, "externalIds": None},
        ],
    },
    None,  # Semantic Scholar doesn't know this paper
]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_scholar, "CACHE_DIR", tmp_path / "s2_cache")
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)


def _response(status_code: int = 200, body: object = SAMPLE_BATCH) -> MagicMock:
    response = MagicMock(status_code=status_code)
    response.json.return_value = body
    return response


def test_fetch_citation_data_parses_batch() -> None:
    with patch("papermesh.semantic_scholar.requests.post", return_value=_response()) as mock_post:
        data = fetch_citation_data(["2301.08243", "hep-th/9901001"])

    assert mock_post.call_args.kwargs["json"] == {"ids": ["ARXIV:2301.08243", "ARXIV:hep-th/9901001"]}
    assert data["hep-th/9901001"] is None
    assert data["2301.08243"] == {
        "s2_id": "s2-jepa",
        "citation_count": 1200,
        "year": 2023,
        "references": [
            {
                "s2_id": "s2-vit",
                "title": "An Image is Worth 16x16 Words",
                "year": 2020,
                "citation_count": 50000,
                "arxiv_id": "2010.11929",
            }
        ],
    }


def test_fetch_citation_data_caches_found_and_missing_papers() -> None:
    with patch("papermesh.semantic_scholar.requests.post", return_value=_response()) as mock_post:
        first = fetch_citation_data(["2301.08243", "hep-th/9901001"])
        second = fetch_citation_data(["2301.08243", "hep-th/9901001"])

    mock_post.assert_called_once()
    assert first == second


def test_fetch_citation_data_retries_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic_scholar, "RETRY_DELAYS_SECONDS", (0, 0))
    responses = [_response(429, None), _response()]
    with patch("papermesh.semantic_scholar.requests.post", side_effect=responses) as mock_post:
        data = fetch_citation_data(["2301.08243", "hep-th/9901001"])

    assert mock_post.call_count == 2
    assert data["2301.08243"]["citation_count"] == 1200


def test_add_citation_data_handles_unknown_papers() -> None:
    papers = [{"id": "2301.08243", "title": "JEPA"}, {"id": "0000.00000", "title": "Unknown"}]
    data = {"2301.08243": {"s2_id": "s2-jepa", "citation_count": 1200, "year": 2023, "references": []}}

    enriched = add_citation_data(papers, data)

    assert enriched[0]["citation_count"] == 1200
    assert enriched[1] == {"id": "0000.00000", "title": "Unknown", "s2_id": None, "citation_count": None, "references": []}
