"""Helpers shared by the Streamlit pages."""

import hashlib
import os
from pathlib import Path

import streamlit as st

from papermesh.library import DEFAULT_DB_PATH, Library


def html(markup: str) -> None:
    # st.markdown instead of st.html: st.html's sanitizer strips target="_blank" from links.
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_resource
def get_library() -> Library:
    # PAPERMESH_DB lets tests and experiments use a throwaway database.
    return Library(Path(os.getenv("PAPERMESH_DB", DEFAULT_DB_PATH)))


def widget_id(value: str) -> str:
    """Short, CSS-safe identifier for widget keys derived from arbitrary strings (paper keys, queries)."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
