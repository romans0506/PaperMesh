"""Page registry, importable from any page that needs to link to another."""

import streamlit as st

from papermesh.views import library, search

SEARCH = st.Page(search.render, title="Search", icon=":material/search:", url_path="search", default=True)
LIBRARY = st.Page(library.render, title="Library", icon=":material/bookmarks:", url_path="library")
PAGES = [SEARCH, LIBRARY]
