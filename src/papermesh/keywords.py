"""Per-paper keyword extraction with TF-IDF over the current result set."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(papers: list[dict], top_n: int = 5) -> list[dict]:
    """Return copies of `papers` with a `keywords` list of their top-N TF-IDF terms."""
    if not papers:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        # Words of 3+ chars starting with a letter; skips numbers and stray symbols.
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([p["abstract"] for p in papers])
    except ValueError:  # empty vocabulary, e.g. all abstracts blank
        return [{**p, "keywords": []} for p in papers]

    terms = vectorizer.get_feature_names_out()
    result = []
    for paper, row in zip(papers, matrix):
        weights = row.toarray().ravel()
        top = [i for i in np.argsort(weights)[::-1][:top_n] if weights[i] > 0]
        result.append({**paper, "keywords": [str(terms[i]) for i in top]})
    return result
