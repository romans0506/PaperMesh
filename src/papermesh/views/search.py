"""Search page: summary, ranked papers, reading order, methodology table, and citation graph."""

from html import escape

import requests
import streamlit as st

from papermesh import components
from papermesh.arxiv_client import search_arxiv
from papermesh.citation_graph import build_citation_graph, result_node_key, shared_foundations, to_graph_data
from papermesh.keywords import extract_keywords
from papermesh.library import paper_key
from papermesh.llm import LLMError, model_name
from papermesh.methodology import cached_extraction, extract_methodology, matches_filters, most_common, table_csv
from papermesh.ranking import (
    DEFAULT_CITATION_WEIGHT,
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_RELEVANCE_WEIGHT,
    rank_papers,
    weighted_rank,
)
from papermesh.reading_order import READING_LENGTHS, build_reading_order
from papermesh.semantic_scholar import add_citation_data, fetch_citation_data
from papermesh.summarizer import summarize_topic
from papermesh.views.common import get_library, html, widget_id

WEIGHT_KEYS = {
    "w_relevance": DEFAULT_RELEVANCE_WEIGHT,
    "w_citations": DEFAULT_CITATION_WEIGHT,
    "w_recency": DEFAULT_RECENCY_WEIGHT,
}
VIEWS = ["Papers", "Reading order", "Methods", "Citation graph"]
METHOD_COUNTS = [5, 10, 20]
GRAPH_HEIGHT_PX = 680


def current_weights() -> tuple[float, float, float]:
    return tuple(st.session_state.get(key, default) for key, default in WEIGHT_KEYS.items())


def run_search(query: str) -> None:
    with st.spinner("Searching arXiv…"):
        try:
            papers = search_arxiv(query)
        except requests.RequestException as e:
            st.error(f"arXiv search failed: {e}")
            return
    if not papers:
        st.info("No papers found for that topic.")
        st.session_state.pop("results", None)
        return

    with st.spinner("Ranking papers…"):
        papers = extract_keywords(rank_papers(query, papers))

    citation_warning = None
    with st.spinner("Fetching citations…"):
        try:
            citation_data = fetch_citation_data([p["id"] for p in papers])
        except requests.RequestException as e:
            citation_data = {}
            citation_warning = f"Semantic Scholar unavailable ({e}); citation counts and graph are missing."
    papers = add_citation_data(papers, citation_data)

    saved_summary = get_library().topic_summary(query)
    if saved_summary:
        summary = saved_summary  # reopening a saved topic: skip regenerating the overview
    else:
        with st.spinner("Writing the overview…"):
            summary = summarize_topic(query, weighted_rank(papers, *current_weights()))

    graph = build_citation_graph(papers)
    graph_data = to_graph_data(graph)
    st.session_state["results"] = {
        "query": query,
        "papers": papers,
        "summary": summary,
        "citation_warning": citation_warning,
        "graph": graph,
        "graph_html": components.citation_graph_html(graph_data) if graph_data["links"] else None,
        "foundations": shared_foundations(graph),
    }


# ---------- Library callbacks ----------

def toggle_topic(query: str, summary: str) -> None:
    library = get_library()
    if library.is_topic_saved(query):
        library.remove_topic(query)
        st.toast("Topic removed from your library", icon=":material/bookmark_remove:")
    else:
        library.save_topic(query, summary)
        st.toast("Topic saved to your library", icon=":material/bookmark_added:")


def toggle_paper(key: str, fields: dict) -> None:
    library = get_library()
    if key in library.saved_paper_keys():
        library.remove_paper(key)
        st.toast("Removed from your library", icon=":material/bookmark_remove:")
    else:
        library.save_paper(key=key, **fields)
        st.toast("Saved to your library", icon=":material/bookmark_added:")


def save_button(key: str, fields: dict, saved_keys: set[str]) -> None:
    saved = key in saved_keys
    st.button(
        "Saved" if saved else "Save",
        icon=":material/bookmark:" if saved else ":material/bookmark_border:",
        key=f"pmsave-{widget_id(key)}",
        type="tertiary",
        help="Remove from library" if saved else "Save to library",
        on_click=toggle_paper,
        args=(key, fields),
    )


# ---------- Sections ----------

def search_bar() -> None:
    with st.form("search", border=False):
        field, button = st.columns([5, 1], vertical_alignment="bottom")
        query = field.text_input(
            "Research topic",
            value=st.query_params.get("q", ""),
            placeholder="Search a research topic, like diffusion models for audio",
            label_visibility="collapsed",
        )
        submitted = button.form_submit_button("Search", type="primary", width="stretch")

    query = query.strip()
    if submitted and query:
        st.query_params["q"] = query
        run_search(query)
    elif not submitted and query and st.session_state.get("results", {}).get("query") != query:
        run_search(query)  # opened via a ?q= link or a saved topic


def ranking_popover() -> None:
    with st.popover("Ranking", icon=":material/tune:"):
        st.caption("Weights are normalized to sum to 1.")
        st.slider("Relevance", 0.0, 1.0, DEFAULT_RELEVANCE_WEIGHT, 0.05, key="w_relevance")
        st.slider("Citations", 0.0, 1.0, DEFAULT_CITATION_WEIGHT, 0.05, key="w_citations")
        st.slider("Recency", 0.0, 1.0, DEFAULT_RECENCY_WEIGHT, 0.05, key="w_recency")


def render_overview(results: dict) -> None:
    saved = get_library().is_topic_saved(results["query"])
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom"):
        html(components.section_html("Overview", results["query"]))
        st.button(
            "Saved topic" if saved else "Save topic",
            icon=":material/bookmark:" if saved else ":material/bookmark_border:",
            key="pm-save-topic",
            on_click=toggle_topic,
            args=(results["query"], results["summary"]),
        )
    html(components.summary_html(results["summary"]))


def render_papers(papers: list[dict], query: str, saved_keys: set[str]) -> None:
    for rank, paper in enumerate(papers, start=1):
        key = paper_key(paper["id"], paper["s2_id"])
        with st.container(key=f"pmcard-{widget_id(key)}"):
            html(components.paper_card_html(rank, paper))
            save_button(key, {
                "title": paper["title"],
                "link": paper["link"],
                "authors": paper["authors"],
                "year": int(paper["published"][:4]),
                "citation_count": paper["citation_count"],
                "source_query": query,
            }, saved_keys)


def render_reading_order(results: dict, papers: list[dict], saved_keys: set[str]) -> None:
    if not results["graph_html"]:
        html(components.summary_html("No citation links found, so there's no reading order to build.", muted=True))
        return

    length = st.segmented_control(
        "Length",
        list(READING_LENGTHS),
        default="Standard",
        required=True,
        label_visibility="collapsed",
        key="reading_length",
    )
    scores = {result_node_key(p): p["score"] for p in papers}
    items = build_reading_order(results["graph"], scores, *READING_LENGTHS[length])
    html(components.reading_summary_html(items))

    stages = [stage for stage in components.STAGES if any(i["stage"] == stage for i in items)]
    for number, stage in enumerate(stages, start=1):
        html(components.stage_header_html(number, stage))
        stage_items = [i for i in items if i["stage"] == stage]
        for position, item in enumerate(stage_items):
            key = paper_key(item["arxiv_id"], item["s2_id"])
            # "pmstepend" marks the last step in a stage, where the timeline line stops.
            kind = "pmstepend" if position == len(stage_items) - 1 else "pmstep"
            with st.container(key=f"{kind}-{widget_id(key)}"):
                html(components.reading_step_html(item))
                save_button(key, {
                    "title": item["title"],
                    "link": item["link"],
                    "year": item["year"],
                    "citation_count": item["citation_count"],
                    "source_query": results["query"],
                }, saved_keys)


def analyze_papers(papers: list[dict]) -> None:
    progress = st.progress(0.0, text="Starting the model…")
    for done, paper in enumerate(papers):
        progress.progress(done / len(papers), text=f"Analyzing {done + 1} of {len(papers)}: {paper['title'][:70]}")
        try:
            extract_methodology(paper)
        except LLMError as e:
            progress.empty()
            st.error(f"Analysis stopped: {e}")
            return
    progress.empty()
    st.rerun()  # redraw the controls now that nothing is missing


def render_methods(papers: list[dict], query: str) -> None:
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        count = st.segmented_control(
            "Papers to compare",
            METHOD_COUNTS,
            format_func=lambda n: f"Top {n}",
            default=10,
            required=True,
            label_visibility="collapsed",
            key="methods_count",
        )
        missing = [p for p in papers[:count] if cached_extraction(p) is None]
        if missing and st.button(
            f"Analyze {len(missing)} paper{'s' if len(missing) != 1 else ''}",
            icon=":material/auto_awesome:",
            type="primary",
            key="pm-analyze",
        ):
            analyze_papers(missing)

    html(
        f'<p class="pm-reading-summary">Extracted from each paper’s title and abstract by {escape(model_name())}. '
        "Only details the abstract actually states are kept, so blanks are common.</p>"
    )

    entries = [(p, row) for p in papers[:count] if (row := cached_extraction(p)) is not None]
    if not entries:
        html(components.empty_state_html(
            "Compare how these papers work",
            f"Analyze the top {count} papers to see each one's task, method, datasets, metrics and key results "
            "side by side. Takes a few seconds per paper the first time; results are saved.",
        ))
        return

    rows = [row for _, row in entries]
    dataset_counts = dict(most_common(rows, "datasets"))
    metric_counts = dict(most_common(rows, "metrics"))
    datasets = st.pills(
        "Datasets", list(dataset_counts), selection_mode="multi", key="methods_datasets",
        format_func=lambda d: f"{d} · {dataset_counts[d]}",
    ) if dataset_counts else []
    metrics = st.pills(
        "Metrics", list(metric_counts), selection_mode="multi", key="methods_metrics",
        format_func=lambda m: f"{m} · {metric_counts[m]}",
    ) if metric_counts else []

    shown = [(p, row) for p, row in entries if matches_filters(row, datasets or [], metrics or [])]
    if not shown:
        html(components.empty_state_html("No matches", "No analyzed paper uses all of the selected datasets and metrics."))
        return
    html(components.methodology_table_html(shown))
    st.download_button(
        "Download CSV",
        data=table_csv(shown),
        file_name=f"papermesh-methods-{widget_id(query)}.csv",
        mime="text/csv",
        icon=":material/download:",
        type="tertiary",
    )


def render_graph(results: dict) -> None:
    if not results["graph_html"]:
        html(components.summary_html("No citation links found between these papers.", muted=True))
        return
    st.iframe(results["graph_html"], height=GRAPH_HEIGHT_PX)

    if results["foundations"]:
        html(components.section_html(
            "Shared foundations",
            "The papers they build on",
            "Cited by several of your results, but not in the results themselves.",
        ))
        html(components.foundations_html(results["foundations"]))


def render() -> None:
    html(components.hero_html())
    search_bar()

    results = st.session_state.get("results")
    if not results:
        return

    if results["citation_warning"]:
        st.warning(results["citation_warning"])

    render_overview(results)

    papers = weighted_rank(results["papers"], *current_weights())
    html(components.section_html("Explore", f"{len(papers)} papers"))
    requested_view = st.query_params.get("view")
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        view = st.segmented_control(
            "View",
            VIEWS,
            default=requested_view if requested_view in VIEWS else VIEWS[0],
            required=True,
            label_visibility="collapsed",
            key="view",
        )
        ranking_popover()

    saved_keys = get_library().saved_paper_keys()
    if view == "Citation graph":
        render_graph(results)
    elif view == "Reading order":
        render_reading_order(results, papers, saved_keys)
    elif view == "Methods":
        render_methods(papers, results["query"])
    else:
        render_papers(papers, results["query"], saved_keys)
