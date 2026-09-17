"""Possible disagreements between papers: claims that contradict or are in tension.

Only disagreements are surfaced. In testing, a small local model reliably spotted tension between
two concrete claims, but labelled almost any two related papers as "agreeing", so agreement flags
were mostly noise and are not produced.

Comparing every pair is too slow for a local model, so only likely-comparable pairs are checked:
papers that share a dataset or metric (from the methodology extraction), or whose abstracts are
very similar. Each pair goes through two passes:

1. find: given both abstracts, the model looks for a conflicting pair of claims and quotes the
   sentence from each abstract that states each side.
2. verify: given only the two quotes, the model checks that both make a claim about the same
   specific question and that the claims really conflict. Short, literal questions like this are
   far more reliable for a small model than judging two whole abstracts.

A flag is kept only if both quotes appear in their abstracts, neither quote is a question or a
background/method description, not both are a paper praising its own model, and the verify pass
confirms the conflict. The judgment can still be wrong, so flags are shown as "possible", with
both quotes side by side for the reader to judge.
"""

import re
from itertools import combinations
from pathlib import Path

import numpy as np

from papermesh import llm
from papermesh.cache import DATA_DIR, read_json, write_json
from papermesh.methodology import name_key

# The cache stores raw replies; validation runs on read. Bump only when a prompt or schema changes.
PROMPT_VERSION = 4
CACHE_DIR = DATA_DIR / "disagreement_cache"
DEFAULT_MAX_PAIRS = 12
MIN_ABSTRACT_SIMILARITY = 0.55
MIN_QUOTE_CHARS = 25
MAX_TEXT_CHARS = 240
# Phrases papers use to praise their own model. Two such claims are not a real conflict.
SELF_PROMOTION_MARKERS = (
    "state-of-the-art", "state of the art", "sota", "outperform", "surpass", "best", "superior",
    "top-ranked", "competitive", "significantly better",
)
# Openings of sentences that describe the paper or its background rather than state a finding.
NON_CLAIM_OPENINGS = (
    "we propose", "we introduce", "we present", "we develop", "we design", "in this paper",
    "in this work", "this paper", "in recent years", "recently",
)

FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "conflict": {"type": "boolean"},
        "topic": {"type": "string"},
        "quote_a": {"type": "string"},
        "quote_b": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["conflict", "topic", "quote_a", "quote_b", "explanation"],
    "additionalProperties": False,
}

FIND_PROMPT = """You look for a disagreement between two research papers, using only their abstracts.

A disagreement is a pair of findings, one from each paper, that contradict or are in tension on the same
specific question: one reports a technique helps where the other reports it hurts or fails, one says
something is slow or costly where the other shows it is fast or cheap, one finds a trade-off the other
finds absent, and so on.

These are NOT disagreements:
- the papers just propose different methods, or study different things;
- each paper says its own model is good ("achieves state-of-the-art results", "outperforms baselines");
- one sentence only describes a method, a dataset, or background.

Example of a real disagreement:
- A: "Diffusion samplers need hundreds of steps to reach high quality."
- B: "We match top quality with only 10 sampling steps."
  (topic: number of sampling steps needed for quality)

If you find one, set conflict to true and:
- quote_a: copy the sentence from Abstract A, word for word, that states A's side.
- quote_b: copy the sentence from Abstract B, word for word, that states B's side.
- topic: what the disagreement is about, in under 10 words.
- explanation: one sentence, under 30 words, on why the two findings conflict.
Otherwise set conflict to false and use empty strings. When in doubt, set conflict to false.

Paper A: {title_a}
Abstract A: {abstract_a}

Paper B: {title_b}
Abstract B: {abstract_b}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "same_question": {"type": "boolean"},
        "conflict": {"type": "boolean"},
    },
    "required": ["question", "same_question", "conflict"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """Two statements come from two different research papers. Read them literally.

Statement A: "{quote_a}"
Statement B: "{quote_b}"

- question: the specific question both statements make a claim about, in under 12 words (empty if none).
- same_question: true only if BOTH statements make a factual claim about that same specific question.
  A statement that only describes a method, poses a question, gives background, or says its own model is
  good makes no claim.
- conflict: true only if the two claims point in opposite directions or cannot both be fully true.
  Two claims that are merely different, or that both say something works, do not conflict."""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _clean_text(value: object) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= MAX_TEXT_CHARS else text[: MAX_TEXT_CHARS - 1].rstrip() + "…"


def _quote_in(quote: str, abstract: str) -> bool:
    """Quote appears in the abstract, ignoring case, spacing and punctuation."""
    normalized = _normalize(quote.rstrip("…"))
    return len(normalized) >= MIN_QUOTE_CHARS and normalized in _normalize(abstract)


def _is_self_promotion(quote: str) -> bool:
    lowered = quote.lower()
    return any(marker in lowered for marker in SELF_PROMOTION_MARKERS)


def _is_claim(quote: str) -> bool:
    lowered = quote.lower().lstrip("\"'( ")
    return not quote.endswith("?") and not lowered.startswith(NON_CLAIM_OPENINGS)


def _quotes(raw: dict) -> tuple[str, str]:
    return " ".join(str(raw.get("quote_a") or "").split()), " ".join(str(raw.get("quote_b") or "").split())


def plausible_disagreement(found: dict, abstract_a: str, abstract_b: str) -> bool:
    """Checks on the find pass that need no model: real, distinct quotes that state claims."""
    if found.get("conflict") is not True:
        return False
    quote_a, quote_b = _quotes(found)
    if not (_quote_in(quote_a, abstract_a) and _quote_in(quote_b, abstract_b)):
        return False  # a quote the model made up or paraphrased: can't be checked by the reader
    if _normalize(quote_a) == _normalize(quote_b):
        return False
    if not (_is_claim(quote_a) and _is_claim(quote_b)):
        return False
    if _is_self_promotion(quote_a) and _is_self_promotion(quote_b):
        return False  # "we are the best" vs "we are the best" is not a conflict
    return True


def validate_disagreement(found: dict, verification: dict | None, abstract_a: str, abstract_b: str) -> dict | None:
    """The verified disagreement, or None if any check fails."""
    if not plausible_disagreement(found, abstract_a, abstract_b) or not verification:
        return None
    if verification.get("same_question") is not True or verification.get("conflict") is not True:
        return None
    quote_a, quote_b = _quotes(found)
    return {
        "topic": _clean_text(found.get("topic")),
        "quote_a": quote_a,
        "quote_b": quote_b,
        "explanation": _clean_text(found.get("explanation")),
        "question": _clean_text(verification.get("question")),
    }


def _shared_names(row_a: dict, row_b: dict) -> list[str]:
    shared = []
    for field in ("datasets", "metrics"):
        keys_b = {name_key(v) for v in row_b[field]}
        shared += [v for v in row_a[field] if name_key(v) in keys_b]
    return shared


def candidate_pairs(
    entries: list[tuple[dict, dict]],
    similarity: np.ndarray,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> list[dict]:
    """Most comparable pairs among (paper, methodology row) entries.

    `similarity[i, j]` is the abstract similarity of entries i and j. Pairs sharing a dataset or
    metric come first; otherwise pairs need abstracts at least MIN_ABSTRACT_SIMILARITY alike.
    """
    pairs = []
    for i, j in combinations(range(len(entries)), 2):
        shared = _shared_names(entries[i][1], entries[j][1])
        sim = float(similarity[i, j])
        if shared or sim >= MIN_ABSTRACT_SIMILARITY:
            pairs.append({"a": entries[i][0], "b": entries[j][0], "shared": shared, "similarity": sim})
    pairs.sort(key=lambda pair: (bool(pair["shared"]), pair["similarity"]), reverse=True)
    return pairs[:max_pairs]


def _cache_path(id_a: str, id_b: str) -> Path:
    model = re.sub(r"[^A-Za-z0-9._-]", "_", f"{llm.backend()}-{llm.model_name()}")
    first, second = sorted([id_a.replace("/", "_"), id_b.replace("/", "_")])
    return CACHE_DIR / model / f"{first}__{second}-v{PROMPT_VERSION}.json"


def _oriented(raw: dict, pair: dict) -> dict:
    """Replies are stored for the pair in sorted-ID order; swap the quotes if this pair is reversed."""
    if pair["a"]["id"] <= pair["b"]["id"]:
        return raw
    return {**raw, "quote_a": raw.get("quote_b"), "quote_b": raw.get("quote_a")}


def _from_cache_entry(entry: dict, pair: dict) -> dict:
    found = _oriented(entry["find"], pair)
    return validate_disagreement(found, entry.get("verify"), pair["a"]["abstract"], pair["b"]["abstract"]) or {}


def cached_check(pair: dict) -> dict | None:
    """None if the pair was never checked; otherwise its verified disagreement, or {} if there is none."""
    entry = read_json(_cache_path(pair["a"]["id"], pair["b"]["id"]), ttl_seconds=None)
    return None if entry is None else _from_cache_entry(entry, pair)


def check_pair(pair: dict) -> dict:
    """Check two papers for a disagreement (or load the cached check).

    Returns the verified disagreement, or {} if none. Raises llm.LLMError.
    """
    cached = cached_check(pair)
    if cached is not None:
        return cached
    a, b = sorted([pair["a"], pair["b"]], key=lambda p: p["id"])
    found = llm.complete_json(
        FIND_PROMPT.format(title_a=a["title"], abstract_a=a["abstract"], title_b=b["title"], abstract_b=b["abstract"]),
        FIND_SCHEMA,
    )
    verify = None
    if plausible_disagreement(found, a["abstract"], b["abstract"]):
        quote_a, quote_b = _quotes(found)
        verify = llm.complete_json(VERIFY_PROMPT.format(quote_a=quote_a, quote_b=quote_b), VERIFY_SCHEMA)
    entry = {"find": found, "verify": verify}
    write_json(_cache_path(a["id"], b["id"]), entry)
    return _from_cache_entry(entry, pair)
