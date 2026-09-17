"""LLM topic summary over the top-ranked papers.

Backends (set SUMMARY_BACKEND in .env):
- "ollama" (default): a local model served by Ollama, free to run.
- "claude": the Anthropic API, needs ANTHROPIC_API_KEY and paid credits.
"""

import os

import anthropic
import requests
from dotenv import load_dotenv

TOP_K = 10

CLAUDE_MODEL = "claude-sonnet-5"
MISSING_KEY_MESSAGE = "Set ANTHROPIC_API_KEY in .env to enable summaries."

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
# Ten abstracts run ~3-5k tokens; Ollama's default context can silently truncate that.
OLLAMA_CONTEXT_TOKENS = 8192
# Generous: the first request after startup includes loading the model into memory.
OLLAMA_TIMEOUT_SECONDS = 300

load_dotenv()


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


def _summarize_with_ollama(prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_ctx": OLLAMA_CONTEXT_TOKENS},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError:
        return (
            f"Summary unavailable: Ollama isn't running at {host}. Install it from "
            f"https://ollama.com and run `ollama pull {model}`."
        )
    except requests.Timeout:
        return "Summary unavailable: the local model took too long to respond."

    if response.status_code == 404:
        return f"Summary unavailable: model '{model}' isn't downloaded. Run `ollama pull {model}`."
    if not response.ok:
        return f"Summary unavailable: Ollama error ({response.status_code})."

    text = response.json().get("message", {}).get("content", "")
    return text.strip() or "Summary unavailable: the model returned no text."


def _summarize_with_claude(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return MISSING_KEY_MESSAGE

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        return "Summary unavailable: ANTHROPIC_API_KEY was rejected. Check the key in .env."
    except anthropic.RateLimitError:
        return "Summary unavailable: rate limited by the Anthropic API. Try again shortly."
    except anthropic.APIStatusError as e:
        return f"Summary unavailable: Anthropic API error ({e.status_code})."
    except anthropic.APIConnectionError:
        return "Summary unavailable: could not reach the Anthropic API."

    if response.stop_reason == "refusal":
        return "Summary unavailable: the model declined to summarize this topic."
    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip() or "Summary unavailable: the model returned no text."


def summarize_topic(query: str, papers: list[dict]) -> str:
    """Synthesize an overview of `query` from the top ranked papers."""
    if not papers:
        return "No papers found to summarize."

    prompt = _build_prompt(query, papers[:TOP_K])
    backend = os.getenv("SUMMARY_BACKEND", "ollama").strip().lower()
    if backend == "claude":
        return _summarize_with_claude(prompt)
    if backend == "ollama":
        return _summarize_with_ollama(prompt)
    return f"Summary unavailable: unknown SUMMARY_BACKEND '{backend}' (use 'ollama' or 'claude')."
