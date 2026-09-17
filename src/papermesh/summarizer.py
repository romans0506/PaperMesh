"""LLM topic summary over the top-ranked papers (backend chosen in papermesh.llm)."""

from papermesh.llm import MISSING_KEY_MESSAGE, LLMError, complete

TOP_K = 10


def _build_prompt(query: str, papers: list[dict]) -> str:
    paper_blocks = "\n\n".join(
        f"[{i}] {p['title']}\n{p['abstract']}" for i, p in enumerate(papers, start=1)
    )
    return (
        f"Below are the titles and abstracts of the {len(papers)} most relevant arXiv "
        f"papers for the research topic \"{query}\".\n\n"
        f"{paper_blocks}\n\n"
        f"Write a 150-250 word plain-language overview of the state of research on "
        f"\"{query}\". Base it only on the papers above: do not add facts, results, or "
        f"papers that are not in them. Cover the main approaches, recurring themes, and "
        f"any open problems the abstracts mention. Write flowing prose with no headings "
        f"or bullet points. The reader can't see the numbered list above, so never refer "
        f"to papers by number (no \"[1]\" or \"paper 3\"); describe the work instead."
    )


def summarize_topic(query: str, papers: list[dict]) -> str:
    """Synthesize an overview of `query` from the top ranked papers."""
    if not papers:
        return "No papers found to summarize."
    try:
        return complete(_build_prompt(query, papers[:TOP_K]))
    except LLMError as e:
        message = str(e)
        return message if message == MISSING_KEY_MESSAGE else f"Summary unavailable: {message}"
