# PaperMesh

A research-literature explorer for ML/CS papers. Search a topic once and get a synthesized
overview, a ranked reading list, a reading order, a methods comparison table, and flags where
papers appear to contradict each other — instead of skimming dozens of PDFs one by one.

Runs locally as a [Streamlit](https://streamlit.io) app. Summaries and extraction run on a local
model via [Ollama](https://ollama.com) by default, so the whole thing is free to run.

## Features

- **Topic overview** — a 150–250 word LLM synthesis of the top 10 results, not per-paper abstracts.
- **Weighted ranking** — semantic relevance (cosine similarity over `all-MiniLM-L6-v2` embeddings)
  blended with log-scaled citation counts and recency (3-year half-life). Weights are adjustable
  in the UI under **Ranking**.
- **Keywords** — top TF-IDF terms per paper, computed over the current result set.
- **Citation graph** — interactive graph of the result papers plus any references shared by two or
  more of them, so the field's foundational work shows up even when it isn't in the results.
- **Reading order** — foundations → core → frontier, with every paper placed after the papers it
  cites. Quick / Standard / Deep lengths.
- **Methods comparison table** — task, method, datasets, metrics and key results extracted per
  paper, with dataset/metric filters and CSV export.
- **Possible disagreements** — closely related pairs are checked for conflicting findings and shown
  as two quotes side by side.
- **Personal library** — save topics (with their overview) and papers from any card or reading-order
  step; track reading status and notes. Stored in local SQLite.

Extraction output is checked before it is shown: methodology names must appear in the abstract and
numbers must match it, and every disagreement is verified word-for-word against both abstracts plus a
second model pass over just the two quotes. Agreements are deliberately not surfaced — in testing the
local model labelled almost any related pair as agreeing.

## Quickstart

Requires Python 3.11+ and a running [Ollama](https://ollama.com).

```bash
git clone <repo-url> && cd PaperMesh
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

ollama pull llama3.1:8b
copy .env.example .env         # cp on macOS/Linux

streamlit run src/papermesh/app.py
```

The app opens at `http://localhost:8501`. Searches are shareable as links:
`http://localhost:8501/?q=diffusion+models+for+audio`.

The first run downloads the sentence-transformers model (~90 MB).

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_BACKEND` | `ollama` | `ollama` (local, free) or `claude` (Anthropic API, paid) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model used for summaries and extraction |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening |
| `ANTHROPIC_API_KEY` | — | Required only when `LLM_BACKEND=claude` (uses `claude-sonnet-5`) |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Optional; only needed if you hit rate limits |
| `PAPERMESH_DB` | `data/papermesh.db` | Point at another file to experiment without touching your library |

## Data sources

- [arXiv API](https://arxiv.org/help/api) — paper metadata and abstracts. Responses cached in
  `data/arxiv_cache/` for 24h.
- [Semantic Scholar API](https://api.semanticscholar.org/) — citation counts and reference lists.
  Cached per paper in `data/s2_cache/` for 24h. Works without an API key.

LLM replies are cached too (`data/methodology_cache/`, `data/disagreement_cache/`), keyed by model,
so re-running a search is fast and doesn't re-spend GPU time. Everything under `data/` is
regenerable and git-ignored.

## Project layout

```
src/papermesh/
  app.py                 Streamlit entry point + top navigation
  views/                 Search and Library pages
  arxiv_client.py        arXiv search (cached)
  semantic_scholar.py    Citation counts + references (cached)
  ranking.py             Relevance / citations / recency blend
  keywords.py            TF-IDF terms per paper
  citation_graph.py      networkx graph of results + shared references
  reading_order.py       Stages + citation-depth ordering
  methodology.py         Per-paper extraction + grounding checks
  consensus.py           Pair selection, find + verify passes
  summarizer.py          Topic overview
  llm.py                 Single entry point for LLM calls (incl. JSON-schema output)
  library.py             SQLite storage (schema versioned via PRAGMA user_version)
  components.py          Paper cards + interactive citation graph
  assets/                app.css, citation_graph.html, force-graph.min.js
tests/                   pytest suite
.streamlit/config.toml   Light theme; file watcher disabled
```

## Development

```bash
pytest
```

Auto-reload on file save is off — the watcher scans every imported module and `transformers`'
optional vision modules make it log ~100 tracebacks on startup. Restart the app after changing code.

## License

MIT — see [LICENSE](LICENSE). Bundles [force-graph](https://github.com/vasturiano/force-graph)
(MIT) in `src/papermesh/assets/`.
