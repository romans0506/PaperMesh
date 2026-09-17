"""Streamlit UI for PaperMesh. Run: streamlit run src/papermesh/app.py"""

import sys
from pathlib import Path

# `streamlit run` puts this file's directory on sys.path, not src/; add src/ so
# the package imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from papermesh import components
from papermesh.arxiv_client import search_arxiv
from papermesh.citation_graph import build_citation_graph, result_node_key, shared_foundations, to_graph_data
from papermesh.keywords import extract_keywords
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

WEIGHT_KEYS = {
    "w_relevance": DEFAULT_RELEVANCE_WEIGHT,
    "w_citations": DEFAULT_CITATION_WEIGHT,
    "w_recency": DEFAULT_RECENCY_WEIGHT,
}
VIEWS = ["Papers", "Reading order", "Citation graph"]
GRAPH_HEIGHT_PX = 680


def html(markup: str) -> None:
    # st.markdown instead of st.html: st.html's sanitizer strips target="_blank" from links.
    st.markdown(markup, unsafe_allow_html=True)


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
        run_search(query)  # opened via a shared ?q= link


def ranking_popover() -> None:
    with st.popover("Ranking", icon=":material/tune:"):
        st.caption("Weights are normalized to sum to 1.")
        st.slider("Relevance", 0.0, 1.0, DEFAULT_RELEVANCE_WEIGHT, 0.05, key="w_relevance")
        st.slider("Citations", 0.0, 1.0, DEFAULT_CITATION_WEIGHT, 0.05, key="w_citations")
        st.slider("Recency", 0.0, 1.0, DEFAULT_RECENCY_WEIGHT, 0.05, key="w_recency")


def render_papers(papers: list[dict]) -> None:
    html(components.paper_list_html(papers))


def render_reading_order(results: dict, papers: list[dict]) -> None:
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
    html(components.reading_order_html(items))


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


def main() -> None:
    st.set_page_config(page_title="PaperMesh", page_icon="📄", layout="wide")
    st.html(components.APP_CSS_PATH)
    html(components.hero_html())
    search_bar()

    results = st.session_state.get("results")
    if not results:
        return

    if results["citation_warning"]:
        st.warning(results["citation_warning"])

    html(components.section_html("Overview", results["query"]))
    html(components.summary_html(results["summary"]))

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

    if view == "Citation graph":
        render_graph(results)
    elif view == "Reading order":
        render_reading_order(results, papers)
    else:
        render_papers(papers)


main()
