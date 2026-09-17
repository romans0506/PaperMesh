"""Citation graph over a result set, built from Semantic Scholar reference lists.

Nodes are the result papers plus "shared foundations": papers outside the results that
several results cite. An edge A -> B means A cites B.
"""

import networkx as nx

DEFAULT_MIN_SHARED_CITATIONS = 2
DEFAULT_MAX_FOUNDATIONS = 15


def result_node_key(paper: dict) -> str:
    return paper["s2_id"] or f"arxiv:{paper['id']}"


def _external_link(ref: dict) -> str:
    if ref["arxiv_id"]:
        return f"https://arxiv.org/abs/{ref['arxiv_id']}"
    return f"https://www.semanticscholar.org/paper/{ref['s2_id']}"


def build_citation_graph(
    papers: list[dict],
    min_shared_citations: int = DEFAULT_MIN_SHARED_CITATIONS,
    max_foundations: int = DEFAULT_MAX_FOUNDATIONS,
) -> nx.DiGraph:
    """Build the graph from papers that have gone through add_citation_data."""
    graph = nx.DiGraph()
    for paper in papers:
        graph.add_node(
            result_node_key(paper),
            title=paper["title"],
            year=int(paper["published"][:4]),
            citation_count=paper.get("citation_count"),
            link=paper["link"],
            in_results=True,
            cited_by_results=0,
        )

    external_refs: dict[str, dict] = {}
    external_citers: dict[str, set[str]] = {}
    for paper in papers:
        citer = result_node_key(paper)
        for ref in paper.get("references", []):
            if ref["s2_id"] == citer:
                continue
            if ref["s2_id"] in graph:
                graph.add_edge(citer, ref["s2_id"])
            else:
                external_refs[ref["s2_id"]] = ref
                external_citers.setdefault(ref["s2_id"], set()).add(citer)

    shared = [key for key, citers in external_citers.items() if len(citers) >= min_shared_citations]
    shared.sort(
        key=lambda key: (len(external_citers[key]), external_refs[key]["citation_count"] or 0),
        reverse=True,
    )
    for key in shared[:max_foundations]:
        ref = external_refs[key]
        graph.add_node(
            key,
            title=ref["title"],
            year=ref["year"],
            citation_count=ref["citation_count"],
            link=_external_link(ref),
            in_results=False,
            cited_by_results=len(external_citers[key]),
        )
        for citer in external_citers[key]:
            graph.add_edge(citer, key)

    for node in graph:
        if graph.nodes[node]["in_results"]:
            graph.nodes[node]["cited_by_results"] = graph.in_degree(node)
    return graph


def shared_foundations(graph: nx.DiGraph) -> list[dict]:
    """Papers outside the result set that several results cite, most-shared first."""
    foundations = [
        {"key": key, **attrs} for key, attrs in graph.nodes(data=True) if not attrs["in_results"]
    ]
    foundations.sort(key=lambda f: (f["cited_by_results"], f["citation_count"] or 0), reverse=True)
    return foundations


def to_graph_data(graph: nx.DiGraph) -> dict:
    """Nodes and links for the interactive graph, skipping papers with no citation links."""
    nodes = [
        {
            "id": key,
            "title": attrs["title"],
            "year": attrs["year"],
            "citations": attrs["citation_count"],
            "link": attrs["link"],
            "inResults": attrs["in_results"],
        }
        for key, attrs in graph.nodes(data=True)
        if graph.degree(key) > 0
    ]
    links = [{"source": citer, "target": cited} for citer, cited in graph.edges]
    return {"nodes": nodes, "links": links}
