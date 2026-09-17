from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from papermesh import consensus
from papermesh.consensus import candidate_pairs, check_pair, plausible_disagreement, validate_disagreement

ABSTRACT_A = (
    "We study chain-of-thought prompting. Experiments on three large language models show that chain of "
    "thought prompting improves performance on a range of arithmetic and commonsense tasks. "
    "Our method achieves state-of-the-art accuracy on GSM8K."
)
ABSTRACT_B = (
    "In recent years, moral reasoning benchmarks have become popular. We propose counterfactual prompting. "
    "Zero-shot chain-of-thought reasoning doesn't work out of the box, and even reduces accuracy by around 4%. "
    "Our approach outperforms all baselines. Can LLMs reason about morality?"
)
QUOTE_A = "chain of thought prompting improves performance on a range of arithmetic and commonsense tasks"
QUOTE_B = "Zero-shot chain-of-thought reasoning doesn't work out of the box, and even reduces accuracy by around 4%."


def _found(**overrides: object) -> dict:
    found = {
        "conflict": True,
        "topic": "whether chain-of-thought helps accuracy",
        "quote_a": QUOTE_A,
        "quote_b": QUOTE_B,
        "explanation": "A finds CoT improves accuracy; B finds zero-shot CoT reduces it.",
    }
    return {**found, **overrides}


VERIFIED = {"question": "does chain-of-thought improve accuracy", "same_question": True, "conflict": True}


def test_valid_disagreement_is_kept() -> None:
    flag = validate_disagreement(_found(), VERIFIED, ABSTRACT_A, ABSTRACT_B)

    assert flag is not None
    assert flag["quote_b"] == QUOTE_B
    assert flag["question"] == "does chain-of-thought improve accuracy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"conflict": False},
        {"quote_b": "Zero-shot chain-of-thought reasoning always helps moral reasoning."},  # not in abstract B
        {"quote_b": "Can LLMs reason about morality?"},  # a question
        {"quote_b": "We propose counterfactual prompting."},  # method description (and too short)
        {"quote_b": "In recent years, moral reasoning benchmarks have become popular."},  # background
        {"quote_a": "Our method achieves state-of-the-art accuracy on GSM8K.", "quote_b": "Our approach outperforms all baselines."},
        {"quote_b": QUOTE_A},  # same sentence on both sides (and not in abstract B)
    ],
)
def test_implausible_disagreements_are_rejected(overrides: dict) -> None:
    assert not plausible_disagreement(_found(**overrides), ABSTRACT_A, ABSTRACT_B)
    assert validate_disagreement(_found(**overrides), VERIFIED, ABSTRACT_A, ABSTRACT_B) is None


@pytest.mark.parametrize(
    "verification",
    [None, {**VERIFIED, "same_question": False}, {**VERIFIED, "conflict": False}, {"question": ""}],
)
def test_verify_pass_must_confirm(verification: dict | None) -> None:
    assert validate_disagreement(_found(), verification, ABSTRACT_A, ABSTRACT_B) is None


def test_quotes_match_despite_spacing_case_and_punctuation() -> None:
    found = _found(quote_a="Chain of thought prompting  improves performance on a range of arithmetic and commonsense tasks.")

    assert plausible_disagreement(found, ABSTRACT_A, ABSTRACT_B)


def _entry(paper_id: str, datasets: list[str], metrics: list[str]) -> tuple[dict, dict]:
    paper = {"id": paper_id, "title": f"Paper {paper_id}", "abstract": f"Abstract {paper_id}"}
    return paper, {"datasets": datasets, "metrics": metrics}


def test_candidate_pairs_prefer_shared_data_then_similarity() -> None:
    entries = [
        _entry("1", ["AudioCaps"], ["FAD"]),
        _entry("2", ["audiocaps"], []),
        _entry("3", [], []),
        _entry("4", [], ["Frechet Audio Distance (FAD)"]),
    ]
    similarity = np.array([
        [1.0, 0.2, 0.9, 0.1],
        [0.2, 1.0, 0.3, 0.1],
        [0.9, 0.3, 1.0, 0.6],
        [0.1, 0.1, 0.6, 1.0],
    ])

    pairs = candidate_pairs(entries, similarity, max_pairs=10)

    ids = [(p["a"]["id"], p["b"]["id"]) for p in pairs]
    assert ids == [("1", "2"), ("1", "4"), ("1", "3"), ("3", "4")]
    assert pairs[0]["shared"] == ["AudioCaps"]
    assert pairs[1]["shared"] == ["FAD"]
    assert ("2", "3") not in ids  # nothing shared and too dissimilar
    assert len(candidate_pairs(entries, similarity, max_pairs=2)) == 2


def test_check_pair_runs_verify_only_when_plausible_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(consensus, "CACHE_DIR", tmp_path)
    paper_a = {"id": "2201.11903", "title": "CoT", "abstract": ABSTRACT_A}
    paper_b = {"id": "2306.14308", "title": "Thought Experiment", "abstract": ABSTRACT_B}

    with patch("papermesh.consensus.llm.complete_json", side_effect=[_found(), VERIFIED]) as mock_llm:
        flag = check_pair({"a": paper_a, "b": paper_b, "shared": [], "similarity": 0.8})
        assert mock_llm.call_count == 2
    assert flag["quote_a"] == QUOTE_A

    # Same pair in the opposite order: served from cache, quotes swapped to match.
    with patch("papermesh.consensus.llm.complete_json") as mock_llm:
        reversed_flag = check_pair({"a": paper_b, "b": paper_a, "shared": [], "similarity": 0.8})
        mock_llm.assert_not_called()
    assert reversed_flag["quote_a"] == QUOTE_B and reversed_flag["quote_b"] == QUOTE_A

    paper_c = {"id": "2305.10601", "title": "Other", "abstract": ABSTRACT_B}
    with patch("papermesh.consensus.llm.complete_json", return_value=_found(conflict=False)) as mock_llm:
        assert check_pair({"a": paper_a, "b": paper_c, "shared": [], "similarity": 0.7}) == {}
        assert mock_llm.call_count == 1  # no verify pass for a non-conflict
    assert consensus.cached_check({"a": paper_a, "b": paper_c}) == {}
    assert consensus.cached_check({"a": paper_b, "b": paper_c}) is None
