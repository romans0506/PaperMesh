from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from papermesh import arxiv_client

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: search_query=all:jepa</title>
  <entry>
    <id>http://arxiv.org/abs/2301.08243v3</id>
    <updated>2023-04-13T17:59:37Z</updated>
    <published>2023-01-19T18:59:01Z</published>
    <title>Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture</title>
    <summary>  This paper demonstrates an approach for learning
highly semantic image representations.
</summary>
    <author><name>Mahmoud Assran</name></author>
    <author><name>Quentin Duval</name></author>
    <link href="http://arxiv.org/abs/2301.08243v3" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.08243v3" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <published>1999-01-01T00:00:00Z</published>
    <title>An old-style identifier</title>
    <summary>Legacy abstract.</summary>
    <author><name>A. Physicist</name></author>
    <link href="http://arxiv.org/abs/hep-th/9901001v1" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arxiv_client, "CACHE_DIR", tmp_path / "arxiv_cache")


def _mock_response() -> MagicMock:
    response = MagicMock()
    response.text = SAMPLE_FEED
    response.raise_for_status.return_value = None
    return response


def test_search_arxiv_parses_response() -> None:
    with patch("papermesh.arxiv_client.requests.get", return_value=_mock_response()) as mock_get:
        papers = arxiv_client.search_arxiv("jepa", max_results=2)

    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["params"]["search_query"] == "all:jepa"
    assert mock_get.call_args.kwargs["params"]["max_results"] == 2

    assert len(papers) == 2
    assert papers[0] == {
        "id": "2301.08243",
        "title": "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
        "abstract": "This paper demonstrates an approach for learning highly semantic image representations.",
        "authors": ["Mahmoud Assran", "Quentin Duval"],
        "published": "2023-01-19T18:59:01Z",
        "link": "http://arxiv.org/abs/2301.08243v3",
    }
    assert papers[1]["id"] == "hep-th/9901001"


def test_search_arxiv_reuses_cache() -> None:
    with patch("papermesh.arxiv_client.requests.get", return_value=_mock_response()) as mock_get:
        first = arxiv_client.search_arxiv("jepa", max_results=2)
        second = arxiv_client.search_arxiv("jepa", max_results=2)

    mock_get.assert_called_once()
    assert first == second
