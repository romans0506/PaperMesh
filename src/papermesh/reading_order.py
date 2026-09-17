"""Reading order: foundational -> advanced, derived from the citation graph.

Papers are grouped into three stages, always read in this order:
- foundations: papers outside the results that several results cite
- core: search results that other results cite
- frontier: search results that nothing else in the set builds on

Foundations and frontier papers are read oldest first: nothing in the set cites a frontier
paper, and foundations' own references aren't known, so neither has ordering constraints.
Core papers can cite each other, so they are sorted by citation depth (a paper's depth is
always greater than everything it cites) before year. Every paper comes after the papers it
cites.
"""

import networkx as nx

STAGE_ORDER = ("foundations", "core", "frontier")
# Preset lengths: (max foundations, max search results).
READING_LENGTHS = {
    "Quick": (3, 5),
    "Standard": (5, 10),
    "Deep": (15, 20),
}
MAX_TITLES_IN_REASON = 1


def citation_depths(graph: nx.DiGraph) -> dict[str, int]:
    """Length of the longest citation chain below each paper (0 = cites nothing in the graph).

    Citation cycles (possible with arXiv revisions) are collapsed so they share one depth.
    """
    condensed = nx.condensation(graph)
    depth: dict[int, int] = {}
    for component in reversed(list(nx.topological_sort(condensed))):
        cited = list(condensed.successors(component))
        depth[component] = 1 + max(depth[c] for c in cited) if cited else 0
    return {node: depth[condensed.graph["mapping"][node]] for node in graph}


def _stage(graph: nx.DiGraph, node: str) -> str:
    if not graph.nodes[node]["in_results"]:
        return "foundations"
    cited_by_results = any(graph.nodes[citer]["in_results"] for citer in graph.predecessors(node))
    return "core" if cited_by_results else "frontier"


def _short_title(title: str, max_chars: int = 64) -> str:
    return title if len(title) <= max_chars else title[: max_chars - 1].rstrip() + "…"


def _reason(graph: nx.DiGraph, node: str, stage: str, reading_list: set[str]) -> str:
    attrs = graph.nodes[node]
    if stage == "foundations":
        count = attrs["cited_by_results"]
        return f"Cited by {count} of your results"
    if stage == "core":
        count = sum(1 for citer in graph.predecessors(node) if graph.nodes[citer]["in_results"])
        return f"Cited by {count} other result{'s' if count != 1 else ''}"

    cited = sorted(
        graph.successors(node),
        # Prefer naming papers the reader will actually see in the list.
        key=lambda n: (
            n in reading_list,
            graph.nodes[n]["cited_by_results"],
            graph.nodes[n]["citation_count"] or 0,
        ),
        reverse=True,
    )
    names = [_short_title(graph.nodes[n]["title"]) for n in cited[:MAX_TITLES_IN_REASON]]
    extra = len(cited) - len(names)
    reason = "Builds on " + " and ".join(names)
    return reason + (f" +{extra} more" if extra > 0 else "")


def build_reading_order(
    graph: nx.DiGraph,
    scores: dict[str, float],
    max_foundations: int = READING_LENGTHS["Standard"][0],
    max_papers: int = READING_LENGTHS["Standard"][1],
) -> list[dict]:
    """Ordered reading list over the most important connected papers.

    `scores` maps result-paper graph keys to their ranking score; it picks which search
    results make the cut. Foundations are picked by how many results cite them.
    Papers with no citation links can't be placed and are left out.
    """
    connected = [n for n in graph if graph.degree(n) > 0]
    foundations = sorted(
        (n for n in connected if not graph.nodes[n]["in_results"]),
        key=lambda n: (graph.nodes[n]["cited_by_results"], graph.nodes[n]["citation_count"] or 0),
        reverse=True,
    )[:max_foundations]
    results = sorted(
        (n for n in connected if graph.nodes[n]["in_results"]),
        key=lambda n: scores.get(n, 0.0),
        reverse=True,
    )[:max_papers]

    depths = citation_depths(graph)
    stages = {n: _stage(graph, n) for n in foundations + results}
    ordered = sorted(
        stages,
        key=lambda n: (
            STAGE_ORDER.index(stages[n]),
            depths[n] if stages[n] == "core" else 0,
            graph.nodes[n]["year"] or 9999,
            -(graph.nodes[n]["citation_count"] or 0),
        ),
    )

    step_of = {node: step for step, node in enumerate(ordered, start=1)}
    selected = set(stages)
    reading_list = []
    for node in ordered:
        attrs = graph.nodes[node]
        reading_list.append({
            "step": step_of[node],
            "key": node,
            "title": attrs["title"],
            "year": attrs["year"],
            "link": attrs["link"],
            "citation_count": attrs["citation_count"],
            "stage": stages[node],
            "reason": _reason(graph, node, stages[node], selected),
            "after_steps": sorted(step_of[cited] for cited in graph.successors(node) if cited in step_of),
        })
    return reading_list
