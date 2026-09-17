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

✅ **Step 1 implemented** — search + LLM topic summary + per-paper keywords + relevance-only ranking.

Summaries run on a local model via [Ollama](https://ollama.com) by default (free):

```bash
ollama pull llama3.1:8b
copy .env.example .env
streamlit run src/papermesh/app.py
pytest
```

To use Claude instead, set `SUMMARY_BACKEND=claude` and `ANTHROPIC_API_KEY` in `.env`
(billed separately via platform.claude.com).

- `src/papermesh/arxiv_client.py` — arXiv search, responses cached in `data/arxiv_cache/` for 24h
- `src/papermesh/ranking.py` — cosine similarity of query vs. abstract (`all-MiniLM-L6-v2`)
- `src/papermesh/keywords.py` — top TF-IDF terms per paper, computed over the current result set
- `src/papermesh/summarizer.py` — 150–250 word overview from the top 10 papers (Ollama, or `claude-sonnet-5`)
- `src/papermesh/app.py` — Streamlit UI

The first run downloads the sentence-transformers model (~90 MB).

Next: step 2 (citation count + recency weighting, citation graph).
