# PaperMesh

An AI agent for exploring ML/CS research literature: search a topic, get a synthesized
summary, ranked and related papers, instead of skimming dozens of PDFs one by one.

## Planned features

- **Search & summary** — search a topic, get an LLM-synthesized overview of the field
- **Keyword extraction** — key terms per paper and per topic
- **Weighted ranking** — combines semantic relevance, citation count, and recency
- **Paper relations** — citation graph + semantic similarity clustering
- **Reading order** — foundational → advanced ordering based on the citation graph
- **Methodology comparison table** — dataset / method / metric / result, extracted per paper
- **Personal library** — save topics and papers for later
- **Consensus / disagreement flags** — surfaces where papers actually contradict each other

## Data sources

- [arXiv API](https://arxiv.org/help/api) — abstracts, full text links
- [Semantic Scholar API](https://api.semanticscholar.org/) — citation graph, semantic search

## Build order

1. Search + summary + keywords + relevance-only ranking
2. Add citation count + recency into a weighted ranking score; build citation graph
3. Reading order (reuses the citation graph)
4. Personal library (first real backend/database work)
5. Methodology comparison table
6. Consensus/disagreement flags

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Status

✅ **Step 1 implemented** — search + LLM topic summary + per-paper keywords + relevance ranking.

✅ **Step 2 implemented** — weighted ranking (relevance + citations + recency, adjustable under "Ranking") and a citation graph with shared foundational papers.

✅ **Step 3 implemented** — reading order: foundations → core → frontier, with every paper placed after
the papers it cites (Quick / Standard / Deep lengths).

✅ **Step 4 implemented** — personal library: save topics (with their overview) and papers from any card or
reading-order step; track reading status and notes on the Library page. Stored locally in SQLite
(`data/papermesh.db`).

✅ **Step 5 implemented** — methodology comparison table: task, method, datasets, metrics and key results
extracted from each paper's abstract by the LLM, with dataset/metric filters and CSV export. Replies are
checked against the abstract (names must appear in it, numbers must match) to filter out invented details.

Summaries run on a local model via [Ollama](https://ollama.com) by default (free):

```bash
ollama pull llama3.1:8b
copy .env.example .env
streamlit run src/papermesh/app.py
pytest
```

To use Claude instead, set `LLM_BACKEND=claude` and `ANTHROPIC_API_KEY` in `.env`
(billed separately via platform.claude.com).

- `src/papermesh/arxiv_client.py` — arXiv search, responses cached in `data/arxiv_cache/` for 24h
- `src/papermesh/semantic_scholar.py` — citation counts + reference lists, cached per paper in `data/s2_cache/` for 24h
- `src/papermesh/ranking.py` — relevance (cosine similarity, `all-MiniLM-L6-v2`) blended with log-scaled citations and recency (3-year half-life)
- `src/papermesh/citation_graph.py` — `networkx` graph of result papers plus references shared by 2+ results
- `src/papermesh/reading_order.py` — stages + citation-depth ordering over the citation graph
- `src/papermesh/library.py` — SQLite storage for saved topics and papers (schema versioned via `PRAGMA user_version`)
- `src/papermesh/keywords.py` — top TF-IDF terms per paper, computed over the current result set
- `src/papermesh/llm.py` — single entry point for LLM calls (Ollama or `claude-sonnet-5`), incl. JSON-schema output
- `src/papermesh/summarizer.py` — 150–250 word overview from the top 10 papers
- `src/papermesh/methodology.py` — per-paper extraction + grounding checks; raw replies cached in `data/methodology_cache/`
- `src/papermesh/app.py` — Streamlit entry point + top navigation
- `src/papermesh/views/` — the Search and Library pages
- `src/papermesh/components.py` + `src/papermesh/assets/` — page styles (`app.css`), paper cards, and the
  interactive citation graph (`citation_graph.html`, built on the bundled MIT-licensed
  [force-graph](https://github.com/vasturiano/force-graph))
- `.streamlit/config.toml` — light theme (Inter font, Apple-style greys and blue)

Search links are shareable: `http://localhost:8501/?q=diffusion+models+for+audio`.

Auto-reload on file save is off (see `.streamlit/config.toml`); restart the app after changing code.
Set `PAPERMESH_DB=path/to/other.db` to try things without touching your real library.

The first run downloads the sentence-transformers model (~90 MB). Semantic Scholar works without an API key; set `SEMANTIC_SCHOLAR_API_KEY` in `.env` if you hit rate limits.

Next: step 6 (consensus / disagreement flags).
