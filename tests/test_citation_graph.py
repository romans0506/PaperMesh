from papermesh.citation_graph import build_citation_graph, shared_foundations, to_graph_data


def _ref(s2_id: str, title: str, citation_count: int = 100, arxiv_id: str | None = None) -> dict:
    return {"s2_id": s2_id, "title": title, "year": 2017, "citation_count": citation_count, "arxiv_id": arxiv_id}


def _paper(arxiv_id: str, s2_id: str | None, references: list[dict]) -> dict:
    return {
        "id": arxiv_id,
        "s2_id": s2_id,
        "title": f"Paper {arxiv_id}",
        "published": "2024-05-01T00:00:00Z",
        "link": f"http://arxiv.org/abs/{arxiv_id}",
        "citation_count": 10,
        "references": references,
    }


PAPERS = [
    _paper("A", "s2-a", [_ref("s2-b", "Paper B"), _ref("s2-attn", "Attention", 90000, "1706.03762"), _ref("s2-rare", "Rare")]),
    _paper("B", "s2-b", [_ref("s2-attn", "Attention", 90000, "1706.03762"), _ref("s2-adam", "Adam", 80000)]),
    _paper("C", "s2-c", [_ref("s2-attn", "Attention", 90000, "1706.03762"), _ref("s2-adam", "Adam", 80000)]),
    _paper("D", None, []),  # not found on Semantic Scholar
]


def test_links_citations_within_results() -> None:
    graph = build_citation_graph(PAPERS)

    assert graph.has_edge("s2-a", "s2-b")
    assert graph.nodes["s2-b"]["cited_by_results"] == 1
    assert "arxiv:D" in graph and graph.degree("arxiv:D") == 0


def test_keeps_only_references_shared_by_several_results() -> None:
    graph = build_citation_graph(PAPERS, min_shared_citations=2)

    assert "s2-rare" not in graph
    foundations = shared_foundations(graph)
    assert [f["key"] for f in foundations] == ["s2-attn", "s2-adam"]
    assert foundations[0]["cited_by_results"] == 3
    assert foundations[0]["link"] == "https://arxiv.org/abs/1706.03762"
    assert foundations[1]["link"] == "https://www.semanticscholar.org/paper/s2-adam"


def test_max_foundations_limits_external_nodes() -> None:
    graph = build_citation_graph(PAPERS, max_foundations=1)

    assert [f["key"] for f in shared_foundations(graph)] == ["s2-attn"]


def test_to_graph_data_skips_isolated_papers() -> None:
    data = to_graph_data(build_citation_graph(PAPERS))

    ids = {node["id"] for node in data["nodes"]}
    assert ids == {"s2-a", "s2-b", "s2-c", "s2-attn", "s2-adam"}
    assert {"source": "s2-a", "target": "s2-b"} in data["links"]
    attention = next(node for node in data["nodes"] if node["id"] == "s2-attn")
    assert attention == {
        "id": "s2-attn",
        "title": "Attention",
        "year": 2017,
        "citations": 90000,
        "link": "https://arxiv.org/abs/1706.03762",
        "inResults": False,
    }
