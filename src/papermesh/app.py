"""PaperMesh Streamlit entry point. Run: streamlit run src/papermesh/app.py"""

import sys
from pathlib import Path

# `streamlit run` puts this file's directory on sys.path, not src/; add src/ so
# the package imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from papermesh import components
from papermesh.views.pages import PAGES

st.set_page_config(page_title="PaperMesh", page_icon="📄", layout="wide")
st.html(components.APP_CSS_PATH)
st.navigation(PAGES, position="top").run()
