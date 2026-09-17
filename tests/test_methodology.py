from pathlib import Path
from unittest.mock import patch

import pytest

from papermesh import methodology
from papermesh.methodology import clean_extraction, extract_methodology, matches_filters, most_common

TITLE = "IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation"
ABSTRACT = (
    "We evaluate on AudioCaps using Frechet Audio Distance (FAD) and objective and subjective evaluation. "
    "Training uses large-scale datasets with high-quality text-audio pairs. "
    "IMPACT reaches FAD 1.07 and runs 5x faster than prior diffusion models."
)


def _raw(**overrides: object) -> dict:
    raw = {
        "contribution": "new method",
        "task": "text-to-audio generation",
        "method": "iterative mask-based parallel decoding",
        "datasets": ["AudioCaps"],
        "metrics": ["Frechet Audio Distance (FAD)"],
        "results": ["FAD 1.07 on AudioCaps"],
    }
    return {**raw, **overrides}


def test_keeps_grounded_names_and_results() -> None:
    row = clean_extraction(_raw(), TITLE, ABSTRACT)

    assert row == _raw()


def test_drops_descriptions_generic_metrics_and_invented_numbers() -> None:
    row = clean_extraction(
        _raw(
            datasets=["AudioCaps", "large-scale datasets with high-quality text-audio pairs", "MusicCaps"],
            metrics=["FAD", "objective and subjective evaluation"],
            results=["FAD 0.95 on AudioCaps", "runs 5x faster than prior diffusion models", "good audio quality"],
        ),
        TITLE,
        ABSTRACT,
    )

    assert row["datasets"] == ["AudioCaps"]  # MusicCaps isn't in the abstract; the other is a description
    assert row["metrics"] == ["FAD"]
    assert row["results"] == ["runs 5x faster than prior diffusion models"]  # 0.95 is invented; no comparison


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("AudioCaps", True),
        ("MUSDB18", True),
        ("GRID corpus", True),
        ("Penn Treebank", True),
        ("Clotho", True),
        ("Audio understanding and reasoning benchmarks", False),
        ("large-scale datasets", False),
        ("speech data", False),
    ],
)
def test_dataset_names_must_look_like_names(name: str, kept: bool) -> None:
    abstract = f"We train and evaluate on {name}."

    row = clean_extraction(_raw(datasets=[name], metrics=[], results=[]), "A title", abstract)

    assert (row["datasets"] == [name]) is kept


def test_generic_metric_words_are_dropped() -> None:
    abstract = "We measure instruction-following skills and word error rate."

    row = clean_extraction(
        _raw(datasets=[], metrics=["instruction-following skills", "word error rate"], results=[]), "T", abstract
    )

    assert row["metrics"] == ["word error rate"]


def test_survey_findings_may_be_qualitative() -> None:
    row = clean_extraction(_raw(contribution="survey", results=["good audio quality"]), TITLE, ABSTRACT)

    assert row["results"] == ["good audio quality"]


def test_unknown_contribution_and_bad_types_are_normalized() -> None:
    row = clean_extraction(
        {"contribution": "breakthrough", "task": None, "method": "x" * 500, "datasets": "AudioCaps"},
        TITLE,
        ABSTRACT,
    )

    assert row["contribution"] == "other"
    assert row["task"] == ""
    assert len(row["method"]) == methodology.MAX_TEXT_CHARS and row["method"].endswith("…")
    assert row["datasets"] == [] and row["metrics"] == [] and row["results"] == []


def test_duplicates_are_removed_and_lists_capped() -> None:
    row = clean_extraction(_raw(datasets=["AudioCaps", "audiocaps", "AUDIOCAPS"]), TITLE, ABSTRACT)

    assert row["datasets"] == ["AudioCaps"]


def test_extract_methodology_caches_per_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(methodology, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "model-a")
    paper = {"id": "2501.00001", "title": TITLE, "abstract": ABSTRACT}

    with patch("papermesh.methodology.llm.complete_json", return_value=_raw()) as mock_llm:
        first = extract_methodology(paper)
        second = extract_methodology(paper)
        assert mock_llm.call_count == 1
        assert first == second == _raw()

        monkeypatch.setenv("OLLAMA_MODEL", "model-b")  # a different model gets its own cache
        extract_methodology(paper)
        assert mock_llm.call_count == 2


def test_cache_stores_raw_reply_and_cleans_on_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(methodology, "CACHE_DIR", tmp_path)
    paper = {"id": "2501.00002", "title": TITLE, "abstract": ABSTRACT}
    raw = _raw(datasets=["AudioCaps", "large-scale datasets with high-quality text-audio pairs"])

    with patch("papermesh.methodology.llm.complete_json", return_value=raw):
        assert extract_methodology(paper)["datasets"] == ["AudioCaps"]

    cached_file = next(tmp_path.rglob("*.json"))
    assert "large-scale datasets" in cached_file.read_text(encoding="utf-8")  # raw reply kept on disk
    assert methodology.cached_extraction(paper)["datasets"] == ["AudioCaps"]  # but cleaned when read


def test_name_key_merges_acronym_variants() -> None:
    assert methodology.name_key("Fréchet Audio Distance (FAD)") == "fad"
    assert methodology.name_key("Fréchet audio distance (FAD) score") == "fad"
    assert methodology.name_key("FAD score") == "fad"
    assert methodology.name_key("MS-COCO") == "mscoco"


def test_most_common_and_filters() -> None:
    rows = [
        _raw(datasets=["AudioCaps", "AudioSet"], metrics=["FAD"]),
        _raw(datasets=["audiocaps"], metrics=["Frechet Audio Distance (FAD)", "IS"]),
        _raw(datasets=["Clotho"], metrics=[]),
    ]

    assert most_common(rows, "datasets")[0] == ("AudioCaps", 2)
    assert most_common(rows, "metrics") == [("FAD", 2), ("IS", 1)]
    assert [matches_filters(r, ["AudioCaps"], ["FAD"]) for r in rows] == [True, True, False]
    assert [matches_filters(r, [], []) for r in rows] == [True, True, True]


def test_table_csv_round_trips() -> None:
    import csv
    import io

    paper = {"title": 'IMPACT, "fast"', "published": "2025-05-01T00:00:00Z", "link": "https://arxiv.org/abs/x"}
    text = methodology.table_csv([(paper, _raw(datasets=["AudioCaps", "Clotho"]))])

    header, row = list(csv.reader(io.StringIO(text)))
    assert header[0] == "title" and row[0] == 'IMPACT, "fast"'
    assert row[1] == "2025"
    assert row[6] == "AudioCaps; Clotho"
