"""Methodology comparison: task / method / datasets / metrics / results extracted per paper.

Extraction reads only the title and abstract. A small local model will sometimes invent,
mislabel, or pad out details, so replies are checked against the text before they are kept:
- datasets must appear in the abstract and look like a name ("AudioCaps", "MUSDB18", "Penn Treebank"),
  not a description such as "large-scale text-audio pairs";
- metrics must appear in the abstract and not be generic ("objective evaluation");
- results must contain a number found in the abstract, or a concrete comparison ("outperforms"),
  except for surveys and analyses, whose findings are qualitative by nature.
"""

import csv
import io
import re
from collections import Counter
from pathlib import Path

from papermesh import llm
from papermesh.cache import DATA_DIR, read_json, write_json

# The cache stores the model's raw reply and cleaning runs on every read, so changing the
# grounding rules needs no re-extraction. Bump PROMPT_VERSION only when the prompt or schema changes.
PROMPT_VERSION = 2
CACHE_DIR = DATA_DIR / "methodology_cache"
CONTRIBUTION_TYPES = ("new method", "dataset or benchmark", "survey", "analysis", "application", "other")
MAX_LIST_ITEMS = {"datasets": 6, "metrics": 6, "results": 3}
MAX_TEXT_CHARS = 140
MAX_NAME_WORDS = 5
GENERIC_METRIC_WORDS = {
    "metric", "metrics", "evaluation", "evaluations", "objective", "subjective",
    "benchmark", "benchmarks", "measure", "measures", "performance", "quality",
    "skill", "skills", "ability", "abilities", "capability", "capabilities",
}
GENERIC_DATASET_PLURALS = {"datasets", "benchmarks", "databases", "corpora", "collections"}
COMPARISON_MARKERS = (
    "outperform", "improv", "better", "faster", "reduc", "increas", "higher", "lower", "fewer",
    "surpass", "state-of-the-art", "sota", "speedup", "speed-up", "compar", "competitive", " than ",
)
QUALITATIVE_CONTRIBUTIONS = {"survey", "analysis"}

SCHEMA = {
    "type": "object",
    "properties": {
        "contribution": {"type": "string", "enum": list(CONTRIBUTION_TYPES)},
        "task": {"type": "string"},
        "method": {"type": "string"},
        "datasets": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "results": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["contribution", "task", "method", "datasets", "metrics", "results"],
    "additionalProperties": False,
}

PROMPT = """You are filling in one row of a table that compares research papers.
Read the paper's title and abstract and extract the fields below.

Rules:
- Use only information explicitly stated in the title or abstract. Never guess or use outside knowledge.
- contribution: the paper's main kind of contribution, one of: {types}.
- task: the problem the paper addresses, in under 10 words.
- method: the approach the paper proposes or studies, in under 12 words. Include the model's name if it has one.
- datasets: names of specific datasets or benchmarks the paper uses, such as "ImageNet" or "AudioCaps".
  Only proper names. Never descriptions of data. Use an empty list if none are named.
- metrics: names of evaluation metrics, such as "FID", "accuracy" or "BLEU". Use an empty list if none are named.
- results: up to 3 key results, each under 15 words. Copy any numbers exactly as written. Use an empty list if
  the abstract states no concrete results.

Title: {title}

Abstract: {abstract}"""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def name_key(text: str) -> str:
    """Identity of a dataset or metric name: its parenthesized acronym if any, minus a trailing "score".

    "Fréchet Audio Distance (FAD)", "FAD score" and "fad" all map to "fad".
    """
    acronym = re.search(r"\(([A-Za-z][A-Za-z0-9-]{1,11})\)", text)
    if acronym:
        return _normalize(acronym.group(1))
    return _normalize(re.sub(r"\bscores?\b", "", text, flags=re.IGNORECASE))


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _clean_text(value: object) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= MAX_TEXT_CHARS else text[: MAX_TEXT_CHARS - 1].rstrip() + "…"


def _looks_like_dataset_name(text: str) -> bool:
    """"AudioCaps", "MUSDB18", "GRID corpus", "Penn Treebank", "Clotho" - not "large-scale audio benchmarks"."""
    words = text.split()
    if len(words) > MAX_NAME_WORDS:
        return False
    if any(re.sub(r"[^a-z]", "", w.lower()) in GENERIC_DATASET_PLURALS for w in words):
        return False
    if any(any(ch.isdigit() for ch in w) or any(ch.isupper() for ch in w[1:]) for w in words):
        return True  # inner capital, digit, or all-caps word
    return all(w[:1].isupper() for w in words)  # Title Case, including single capitalized words


def _is_grounded(field: str, text: str, source: str, qualitative_results: bool = False) -> bool:
    words = text.split()
    if field == "datasets":
        return _normalize(text) in _normalize(source) and _looks_like_dataset_name(text)
    if field == "metrics":
        return (
            _normalize(text) in _normalize(source)
            and len(words) <= MAX_NAME_WORDS + 1
            and not any(re.sub(r"[^a-z]", "", w.lower()) in GENERIC_METRIC_WORDS for w in words)
        )
    # results
    numbers = _numbers(text)
    if numbers:
        return numbers <= _numbers(source)
    return qualitative_results or any(marker in f" {text.lower()} " for marker in COMPARISON_MARKERS)


def _clean_list(values: object, field: str, source: str, qualitative_results: bool = False) -> list[str]:
    if not isinstance(values, list):
        return []
    kept: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = name_key(text) if field in ("datasets", "metrics") else _normalize(text)
        if key and key not in seen and _is_grounded(field, text, source, qualitative_results):
            seen.add(key)
            kept.append(text)
    return kept[: MAX_LIST_ITEMS[field]]


def clean_extraction(raw: dict, title: str, abstract: str) -> dict:
    """Validate a model reply against the paper text; drops anything not grounded in it."""
    source = f"{title}\n{abstract}"
    contribution = raw.get("contribution")
    contribution = contribution if contribution in CONTRIBUTION_TYPES else "other"
    return {
        "contribution": contribution,
        "task": _clean_text(raw.get("task")),
        "method": _clean_text(raw.get("method")),
        "datasets": _clean_list(raw.get("datasets"), "datasets", source),
        "metrics": _clean_list(raw.get("metrics"), "metrics", source),
        "results": _clean_list(raw.get("results"), "results", source, contribution in QUALITATIVE_CONTRIBUTIONS),
    }


def _cache_path(paper_id: str) -> Path:
    model = re.sub(r"[^A-Za-z0-9._-]", "_", f"{llm.backend()}-{llm.model_name()}")
    safe_id = paper_id.replace("/", "_")
    return CACHE_DIR / model / f"{safe_id}-v{PROMPT_VERSION}.json"


def cached_extraction(paper: dict) -> dict | None:
    """A previous extraction for this paper with the current backend, model and prompt, cleaned."""
    raw = read_json(_cache_path(paper["id"]), ttl_seconds=None)
    return None if raw is None else clean_extraction(raw, paper["title"], paper["abstract"])


def extract_methodology(paper: dict) -> dict:
    """Extract (or load from cache) the methodology row for one paper. Raises llm.LLMError."""
    cached = cached_extraction(paper)
    if cached is not None:
        return cached
    prompt = PROMPT.format(types=", ".join(CONTRIBUTION_TYPES), title=paper["title"], abstract=paper["abstract"])
    raw = llm.complete_json(prompt, SCHEMA)
    write_json(_cache_path(paper["id"]), raw)
    return clean_extraction(raw, paper["title"], paper["abstract"])


def most_common(rows: list[dict], field: str, limit: int = 8) -> list[tuple[str, int]]:
    """Most frequent datasets or metrics across rows, merging variants of the same name (see name_key)."""
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for row in rows:
        for key, text in {name_key(v): v for v in row[field]}.items():  # count each paper once
            counts[key] += 1
            display.setdefault(key, text)
    return [(display[key], count) for key, count in counts.most_common(limit)]


def matches_filters(row: dict, datasets: list[str], metrics: list[str]) -> bool:
    """True if the row uses every selected dataset and metric (case/punctuation-insensitive)."""
    row_datasets = {name_key(d) for d in row["datasets"]}
    row_metrics = {name_key(m) for m in row["metrics"]}
    return all(name_key(d) in row_datasets for d in datasets) and all(name_key(m) in row_metrics for m in metrics)


def table_csv(entries: list[tuple[dict, dict]]) -> str:
    """CSV export of (paper, extraction row) pairs; list fields are joined with "; "."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["title", "year", "link", "contribution", "task", "method", "datasets", "metrics", "results"])
    for paper, row in entries:
        writer.writerow([
            paper["title"],
            paper["published"][:4],
            paper["link"],
            row["contribution"],
            row["task"],
            row["method"],
            "; ".join(row["datasets"]),
            "; ".join(row["metrics"]),
            "; ".join(row["results"]),
        ])
    return buffer.getvalue()
