import json

from papermesh import components

PAPER = {
    "title": "Attention <Is> All You Need",
    "link": "http://arxiv.org/abs/1706.03762v7",
    "authors": ["A. Vaswani", "N. Shazeer", "N. Parmar", "J. Uszkoreit", "L. Jones"],
    "published": "2017-06-12T17:57:34Z",
    "abstract": "The dominant sequence transduction models...",
    "keywords": ["attention", "transformer"],
    "citation_count": 1,
    "score": 0.8,
    "relevance_score": 0.9,
    "citation_score": 0.5,
    "recency_score": 0.25,
}


def test_paper_card_escapes_and_formats() -> None:
    card = components.paper_card_html(3, PAPER)

    assert "Attention &lt;Is&gt; All You Need" in card
    assert "<Is>" not in card
    assert "A. Vaswani, N. Shazeer, N. Parmar +2 more" in card
    assert "2017 · 1 citation<" in card
    assert 'target="_blank"' in card
    assert '<span class="pm-rank">03</span>' in card
    assert "\n" not in card  # blank lines would break the markdown HTML block


def test_citation_graph_html_embeds_data_safely() -> None:
    data = {"nodes": [{"id": "x", "title": "</script><b>evil</b>"}], "links": []}

    page = components.citation_graph_html(data)

    assert "/*__GRAPH_DATA__*/" not in page and "/*__FORCE_GRAPH_JS__*/" not in page
    assert "</script><b>evil" not in page
    embedded = page.split("const data = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(embedded) == data  # "<\/" is a valid JSON escape for "</"


def test_reading_step_and_summary_html() -> None:
    items = [
        {"step": 1, "key": "f", "title": "DDPM", "year": 2020, "link": "https://arxiv.org/abs/2006.11239",
         "citation_count": 35007, "stage": "foundations", "reason": "Cited by 10 of your results", "after_steps": []},
        {"step": 2, "key": "r", "title": "Audio <LDM>", "year": 2023, "link": "https://arxiv.org/abs/2301.12503",
         "citation_count": None, "stage": "frontier", "reason": "Builds on DDPM", "after_steps": [1]},
    ]

    assert "2 papers · 1 foundations · 1 frontier" in components.reading_summary_html(items)
    step = components.reading_step_html(items[1])
    assert "Audio &lt;LDM&gt;" in step
    assert "After step 1<" in step
    assert "pm-step-frontier" in step
    assert "35,007 citations" in components.reading_step_html(items[0])


def test_saved_paper_and_topic_html() -> None:
    paper = {
        "key": "s2:x", "title": "AudioSet", "link": "https://example.org", "authors": [], "year": None,
        "citation_count": 4243, "source_query": "audio <gen>", "status": "reading", "note": "Check table 2",
    }
    html = components.saved_paper_html(paper)
    assert "Reading" in html and "Year unknown · 4,243 citations" in html
    assert "audio &lt;gen&gt;" in html
    assert "Check table 2" in html
    assert "pm-authors" not in html  # no empty author line

    topic = {"display": "Diffusion", "summary": "word " * 100, "saved_at": "2026-09-17T10:00:00+00:00", "paper_count": 1}
    html = components.topic_card_html(topic)
    assert "Saved Sep 17, 2026 · 1 saved paper<" in html
    assert "…" in html


def test_methodology_table_html_escapes_and_marks_missing() -> None:
    paper = {"title": "A <b>paper</b>", "link": "https://arxiv.org/abs/1", "published": "2024-01-01T00:00:00Z"}
    row = {
        "contribution": "new method", "task": "", "method": "Model X",
        "datasets": ["AudioCaps"], "metrics": [], "results": ["FAD 1.2"],
    }

    table = components.methodology_table_html([(paper, row)])

    assert "A &lt;b&gt;paper&lt;/b&gt;" in table
    assert '<span class="pm-chip ">AudioCaps</span>' in table
    assert "Metrics" not in table  # empty sections are omitted
    no_data = components.methodology_table_html([(paper, {**row, "datasets": []})])
    assert '<td class="pm-col-data"><span class="pm-none">—</span></td>' in no_data
    assert "<li>FAD 1.2</li>" in table
    assert "pm-task" not in table  # empty task is omitted
    assert "\n" not in table


def test_disagreement_card_html() -> None:
    paper_a = {"title": "CoT <works>", "link": "https://arxiv.org/abs/a", "published": "2022-01-28T00:00:00Z"}
    paper_b = {"title": "Thought Experiment", "link": "https://arxiv.org/abs/b", "published": "2023-06-25T00:00:00Z"}
    flag = {
        "topic": "", "question": "does CoT help accuracy", "explanation": "They conflict.",
        "quote_a": "CoT improves accuracy.", "quote_b": "CoT reduces accuracy by 4%.",
    }

    card = components.disagreement_card_html(flag, {"a": paper_a, "b": paper_b, "shared": ["GSM8K"]})

    assert "Possible disagreement" in card
    assert "<h3>does CoT help accuracy</h3>" in card  # falls back to the verified question
    assert "CoT &lt;works&gt;" in card
    assert card.index("CoT improves accuracy.") < card.index("CoT reduces accuracy by 4%.")
    assert "Both use" in card and "GSM8K" in card
    assert "\n" not in card
