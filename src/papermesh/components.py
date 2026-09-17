"""HTML building blocks for the Streamlit UI (styles live in assets/app.css)."""

import json
from functools import lru_cache
from html import escape
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
APP_CSS_PATH = ASSETS_DIR / "app.css"
MAX_AUTHORS_SHOWN = 3
# Reading-order stage -> (heading, description).
STAGES = {
    "foundations": ("Foundations", "The groundwork your results keep citing. Start here."),
    "core": ("Core", "Search results that other results build on."),
    "frontier": ("Frontier", "Where the field is now, oldest to newest."),
}


@lru_cache(maxsize=1)
def _graph_template() -> str:
    template = (ASSETS_DIR / "citation_graph.html").read_text(encoding="utf-8")
    library = (ASSETS_DIR / "force-graph.min.js").read_text(encoding="utf-8")
    return template.replace("/*__FORCE_GRAPH_JS__*/", library)


def hero_html() -> str:
    return (
        '<div class="pm-hero">'
        '<p class="pm-eyebrow">PaperMesh</p>'
        "<h1>Research, untangled.</h1>"
        '<p class="pm-sub">Search arXiv. See what matters, and how it all connects.</p>'
        "</div>"
    )


def section_html(eyebrow: str, title: str, note: str | None = None) -> str:
    note_html = f'<p class="pm-note">{escape(note)}</p>' if note else ""
    return (
        f'<div class="pm-section"><p class="pm-eyebrow">{escape(eyebrow)}</p>'
        f"<h2>{escape(title)}</h2>{note_html}</div>"
    )


def summary_html(summary: str, muted: bool = False) -> str:
    paragraphs = [p.strip() for p in summary.split("\n\n") if p.strip()]
    body = "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
    return f'<div class="pm-summary{" pm-muted" if muted else ""}">{body}</div>'


def _authors(authors: list[str]) -> str:
    shown = ", ".join(authors[:MAX_AUTHORS_SHOWN])
    extra = len(authors) - MAX_AUTHORS_SHOWN
    return f"{shown} +{extra} more" if extra > 0 else shown


def _bar(css_class: str, label: str, value: float) -> str:
    return (
        f'<div><div class="pm-bar-label">{label}<b>{value:.2f}</b></div>'
        f'<div class="pm-bar {css_class}"><span style="width:{value * 100:.0f}%"></span></div></div>'
    )


def paper_card_html(rank: int, paper: dict) -> str:
    citations = paper.get("citation_count")
    meta = [paper["published"][:4]]
    if citations is not None:
        meta.append(f"{citations:,} citation{'s' if citations != 1 else ''}")
    chips = "".join(f'<span class="pm-chip">{escape(k)}</span>' for k in paper["keywords"])
    return (
        '<article class="pm-card">'
        f'<div class="pm-card-top"><span class="pm-rank">{rank:02d}</span>'
        f'<span class="pm-meta">{" · ".join(meta)}</span>'
        f'<span class="pm-score">{paper["score"]:.2f}</span></div>'
        f'<h3><a href="{escape(paper["link"])}" target="_blank" rel="noopener">{escape(paper["title"])}</a></h3>'
        f'<p class="pm-authors">{escape(_authors(paper["authors"]))}</p>'
        '<div class="pm-bars">'
        f'{_bar("pm-rel", "Relevance", paper["relevance_score"])}'
        f'{_bar("pm-cit", "Citations", paper["citation_score"])}'
        f'{_bar("pm-rec", "Recency", paper["recency_score"])}'
        "</div>"
        f'<div class="pm-chips">{chips}</div>'
        f'<details><summary>Abstract</summary><p>{escape(paper["abstract"])}</p></details>'
        "</article>"
    )


def paper_list_html(papers: list[dict]) -> str:
    cards = "".join(paper_card_html(rank, paper) for rank, paper in enumerate(papers, start=1))
    return f'<div class="pm-list">{cards}</div>'


def foundations_html(foundations: list[dict]) -> str:
    cards = []
    for f in foundations:
        meta = [str(f["year"]) if f["year"] else "Year unknown"]
        if f["citation_count"] is not None:
            meta.append(f"{f['citation_count']:,} citations")
        cards.append(
            f'<a class="pm-foundation" href="{escape(f["link"])}" target="_blank" rel="noopener">'
            f'<div class="pm-f-count">Cited by {f["cited_by_results"]} results</div>'
            f'<div class="pm-f-title">{escape(f["title"])}</div>'
            f'<div class="pm-f-meta">{" · ".join(meta)}</div></a>'
        )
    return f'<div class="pm-foundations">{"".join(cards)}</div>'


def citation_graph_html(graph_data: dict) -> str:
    # "</" inside a <script> would end it early if a title contained "</script>".
    payload = json.dumps(graph_data, ensure_ascii=False).replace("</", "<\\/")
    return _graph_template().replace("/*__GRAPH_DATA__*/", payload)


def _reading_step_html(item: dict) -> str:
    meta = [str(item["year"]) if item["year"] else "Year unknown"]
    if item["citation_count"] is not None:
        meta.append(f"{item['citation_count']:,} citations")
    after = ""
    if item["after_steps"]:
        label = "step" if len(item["after_steps"]) == 1 else "steps"
        after = f'<div class="pm-step-after">After {label} {", ".join(map(str, item["after_steps"]))}</div>'
    return (
        f'<li class="pm-step pm-step-{item["stage"]}">'
        '<span class="pm-dot"></span>'
        '<div class="pm-step-body">'
        f'<div class="pm-step-top"><span class="pm-step-num">Step {item["step"]}</span>'
        f'<span>{" · ".join(meta)}</span></div>'
        f'<a class="pm-step-title" href="{escape(item["link"])}" target="_blank" rel="noopener">'
        f'{escape(item["title"])}</a>'
        f'<div class="pm-step-reason">{escape(item["reason"])}</div>'
        f"{after}"
        "</div></li>"
    )


def reading_order_html(items: list[dict]) -> str:
    sections = []
    for number, stage in enumerate((s for s in STAGES if any(i["stage"] == s for i in items)), start=1):
        title, description = STAGES[stage]
        steps = "".join(_reading_step_html(i) for i in items if i["stage"] == stage)
        sections.append(
            '<section class="pm-stage">'
            f'<div class="pm-stage-head"><span class="pm-stage-badge">{number}</span>'
            f"<div><h3>{title}</h3><p>{description}</p></div></div>"
            f'<ol class="pm-steps">{steps}</ol>'
            "</section>"
        )
    counts = [
        f"{sum(1 for i in items if i['stage'] == stage)} {title.lower()}"
        for stage, (title, _) in STAGES.items()
        if any(i["stage"] == stage for i in items)
    ]
    summary = f'<p class="pm-reading-summary">{len(items)} papers · {" · ".join(counts)}</p>'
    return f'<div class="pm-reading">{summary}{"".join(sections)}</div>'
