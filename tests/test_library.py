import sqlite3
from pathlib import Path

import pytest

from papermesh.library import SCHEMA_VERSION, Library, paper_key


@pytest.fixture
def library(tmp_path: Path) -> Library:
    return Library(tmp_path / "test.db")


def _save_ddpm(library: Library, **overrides: object) -> None:
    fields = {
        "key": "arxiv:2006.11239",
        "title": "Denoising Diffusion Probabilistic Models",
        "link": "https://arxiv.org/abs/2006.11239",
        "authors": ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
        "year": 2020,
        "citation_count": 35000,
        "source_query": "Diffusion Models",
    }
    library.save_paper(**{**fields, **overrides})


def test_paper_key_prefers_arxiv() -> None:
    assert paper_key("2006.11239", "abc") == "arxiv:2006.11239"
    assert paper_key(None, "abc") == "s2:abc"
    with pytest.raises(ValueError):
        paper_key(None, None)


def test_schema_version_is_set(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    Library(db)
    Library(db)  # reopening must not re-run migrations

    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_topics_are_saved_normalized_and_listed_newest_first(library: Library) -> None:
    library.save_topic("  Diffusion   Models ", summary="An overview.")
    library.save_topic("graph neural networks")

    assert library.is_topic_saved("diffusion models")
    topics = library.list_topics()
    assert [t["display"] for t in topics] == ["graph neural networks", "Diffusion Models"]
    assert topics[1]["summary"] == "An overview."

    assert library.topic_summary("DIFFUSION MODELS") == "An overview."
    assert library.topic_summary("graph neural networks") is None
    assert library.topic_summary("never saved") is None

    library.remove_topic("DIFFUSION MODELS")
    assert not library.is_topic_saved("diffusion models")


def test_topic_counts_papers_saved_from_it(library: Library) -> None:
    library.save_topic("Diffusion Models")
    _save_ddpm(library)

    assert library.list_topics()[0]["paper_count"] == 1


def test_save_list_and_remove_papers(library: Library) -> None:
    _save_ddpm(library)
    library.save_paper(key="s2:xyz", title="AudioSet", link="https://example.org/audioset")

    papers = library.list_papers()
    assert [p["key"] for p in papers] == ["s2:xyz", "arxiv:2006.11239"]
    assert papers[1]["authors"] == ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"]
    assert papers[1]["status"] == "to_read"
    assert library.saved_paper_keys() == {"s2:xyz", "arxiv:2006.11239"}

    library.remove_paper("s2:xyz")
    assert library.saved_paper_keys() == {"arxiv:2006.11239"}


def test_resaving_keeps_status_note_and_known_metadata(library: Library) -> None:
    _save_ddpm(library)
    library.set_status("arxiv:2006.11239", "reading")
    library.set_note("arxiv:2006.11239", "Read section 3 first.")

    # Saved again from the reading order, which has no authors and a fresher citation count.
    _save_ddpm(library, authors=[], year=None, citation_count=36000)

    paper = library.list_papers()[0]
    assert paper["status"] == "reading"
    assert paper["note"] == "Read section 3 first."
    assert paper["authors"] == ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"]
    assert paper["year"] == 2020
    assert paper["citation_count"] == 36000


def test_status_filter_counts_and_validation(library: Library) -> None:
    _save_ddpm(library)
    library.save_paper(key="s2:xyz", title="AudioSet", link="https://example.org/audioset")
    library.set_status("s2:xyz", "read")

    assert [p["key"] for p in library.list_papers(status="read")] == ["s2:xyz"]
    assert library.status_counts() == {"to_read": 1, "reading": 0, "read": 1}
    with pytest.raises(ValueError):
        library.set_status("s2:xyz", "done")
    with pytest.raises(ValueError):
        library.list_papers(status="done")
