"""Streamlit UI for PaperMesh step 1. Run: streamlit run src/papermesh/app.py"""

import sys
from pathlib import Path

# `streamlit run` puts this file's directory on sys.path, not src/; add src/ so
# the package imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from papermesh.arxiv_client import search_arxiv
from papermesh.keywords import extract_keywords
from papermesh.ranking import rank_papers
from papermesh.summarizer import summarize_topic


def render_paper(paper: dict) -> None:
    label = f"{paper['relevance_score']:.2f} · {paper['title']}"
    with st.expander(label):
        st.markdown(f"**{paper['title']}**")
        st.write(f"**Authors:** {', '.join(paper['authors'])}")
        st.write(f"**Published:** {paper['published'][:10]}")
        st.write(f"**Relevance score:** {paper['relevance_score']:.2f}")
        st.write(f"**Keywords:** {', '.join(paper['keywords'])}")
        st.markdown(f"[View on arXiv]({paper['link']})")


def main() -> None:
    st.set_page_config(page_title="PaperMesh", page_icon="📄")
    st.title("PaperMesh")

    query = st.text_input("Research topic", placeholder="e.g. diffusion models for audio")
    if not st.button("Search", type="primary") or not query.strip():
        return

    with st.spinner("Searching arXiv..."):
        try:
            papers = search_arxiv(query)
        except requests.RequestException as e:
            st.error(f"arXiv search failed: {e}")
            return
    if not papers:
        st.info("No papers found for that query.")
        return

    with st.spinner("Ranking papers..."):
        papers = extract_keywords(rank_papers(query, papers))
    with st.spinner("Summarizing the topic..."):
        summary = summarize_topic(query, papers)

    st.subheader("Topic summary")
    st.write(summary)

    st.subheader(f"Papers ({len(papers)})")
    for paper in papers:
        render_paper(paper)


main()
