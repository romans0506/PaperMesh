import networkx as nx

from papermesh.reading_order import build_reading_order, citation_depths


def _node(graph: nx.DiGraph, key: str, year: int, in_results: bool, citations: int = 10) -> None:
    graph.add_node(
        key,
        title=f"Paper {key}",
        year=year,
        citation_count=citations,
        link=f"https://example.org/{key}",
        in_results=in_results,
        cited_by_results=0,
    )


def _graph() -> nx.DiGraph:
    """F1, F2 are foundations; C1 <- C2 are core (C2 cites C1); R1, R2 are frontier."""
    g = nx.DiGraph()
    _node(g, "F1", 2015, False, citations=50000)
    _node(g, "F2", 2012, False, citations=90000)
    _node(g, "C1", 2019, True)
    _node(g, "C2", 2018, True)  # older than C1 but cites it (e.g. an updated arXiv version)
    _node(g, "R1", 2024, True)
    _node(g, "R2", 2021, True)
    _node(g, "LONE", 2020, True)  # no citation links at all
    g.add_edges_from([
        ("C1", "F1"), ("C1", "F2"),
        ("C2", "C1"), ("C2", "F1"),
        ("R1", "C2"), ("R1", "F1"),
        ("R2", "C1"), ("R2", "F2"),
    ])
    g.nodes["F1"]["cited_by_results"] = 3
    g.nodes["F2"]["cited_by_results"] = 2
    return g


SCORES = {"C1": 0.5, "C2": 0.6, "R1": 0.9, "R2": 0.7, "LONE": 1.0}


def test_citation_depths_follow_longest_chain() -> None:
    depths = citation_depths(_graph())

    assert depths["F1"] == 0 and depths["F2"] == 0
    assert depths["C1"] == 1
    assert depths["C2"] == 2
    assert depths["R1"] == 3


def test_citation_depths_handle_cycles() -> None:
    g = nx.DiGraph([("A", "B"), ("B", "A"), ("C", "A")])

    depths = citation_depths(g)

    assert depths["A"] == depths["B"] == 0
    assert depths["C"] == 1


def test_reading_order_puts_every_paper_after_what_it_cites() -> None:
    graph = _graph()
    order = build_reading_order(graph, SCORES, max_foundations=5, max_papers=10)

    steps = {item["key"]: item["step"] for item in order}
    assert [item["key"] for item in order] == ["F2", "F1", "C1", "C2", "R2", "R1"]
    for citer, cited in graph.edges:
        assert steps[cited] < steps[citer]
    assert "LONE" not in steps  # unconnected papers can't be placed


def test_reading_order_stages_reasons_and_after_steps() -> None:
    order = {item["key"]: item for item in build_reading_order(_graph(), SCORES)}

    assert order["F1"]["stage"] == "foundations"
    assert order["F1"]["reason"] == "Cited by 3 of your results"
    assert order["C1"]["stage"] == "core"
    assert order["C1"]["reason"] == "Cited by 2 other results"
    assert order["R1"]["stage"] == "frontier"
    assert order["R1"]["reason"] == "Builds on Paper F1 +1 more"
    assert order["C2"]["after_steps"] == [2, 3]


def test_reading_order_respects_length_limits() -> None:
    order = build_reading_order(_graph(), SCORES, max_foundations=1, max_papers=2)

    assert [item["key"] for item in order] == ["F1", "R2", "R1"]
