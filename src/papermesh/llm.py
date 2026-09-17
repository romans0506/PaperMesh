"""One entry point for LLM calls, backed by a local Ollama model (default) or the Claude API.

Set LLM_BACKEND in .env to "ollama" or "claude" (SUMMARY_BACKEND is still accepted).
"""

import json
import os

import anthropic
import requests
from dotenv import load_dotenv

CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
# Prompts here run ~3-5k tokens; Ollama's default context can silently truncate that.
OLLAMA_CONTEXT_TOKENS = 8192
# Generous: the first request after startup includes loading the model into memory.
OLLAMA_TIMEOUT_SECONDS = 300
MISSING_KEY_MESSAGE = "Set ANTHROPIC_API_KEY in .env to use the Claude backend."

load_dotenv()


class LLMError(Exception):
    """A failure with a message that can be shown to the user as-is."""


def backend() -> str:
    return (os.getenv("LLM_BACKEND") or os.getenv("SUMMARY_BACKEND") or "ollama").strip().lower()


def model_name() -> str:
    if backend() == "claude":
        return CLAUDE_MODEL
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def complete(prompt: str, schema: dict | None = None) -> str:
    """Send a single-turn prompt and return the text reply.

    With `schema`, the reply is constrained to JSON matching that JSON Schema (returned as a string).
    Raises LLMError with a user-facing message on any failure.
    """
    name = backend()
    if name == "ollama":
        return _complete_ollama(prompt, schema)
    if name == "claude":
        return _complete_claude(prompt, schema)
    raise LLMError(f"Unknown LLM_BACKEND '{name}' (use 'ollama' or 'claude').")


def _complete_ollama(prompt: str, schema: dict | None) -> str:
    host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    model = model_name()
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": OLLAMA_CONTEXT_TOKENS},
    }
    if schema is not None:
        body["format"] = schema
        body["options"]["temperature"] = 0  # extraction should be repeatable
    try:
        response = requests.post(f"{host}/api/chat", json=body, timeout=OLLAMA_TIMEOUT_SECONDS)
    except requests.ConnectionError:
        raise LLMError(
            f"Ollama isn't running at {host}. Install it from https://ollama.com and run `ollama pull {model}`."
        ) from None
    except requests.Timeout:
        raise LLMError("the local model took too long to respond.") from None

    if response.status_code == 404:
        raise LLMError(f"model '{model}' isn't downloaded. Run `ollama pull {model}`.")
    if not response.ok:
        raise LLMError(f"Ollama error ({response.status_code}).")

    text = response.json().get("message", {}).get("content", "").strip()
    if not text:
        raise LLMError("the model returned no text.")
    return text


def _complete_claude(prompt: str, schema: dict | None) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError(MISSING_KEY_MESSAGE)

    params: dict = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": "low"},
    }
    if schema is not None:
        params["output_config"]["format"] = {"type": "json_schema", "schema": schema}

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(**params)
    except anthropic.AuthenticationError:
        raise LLMError("ANTHROPIC_API_KEY was rejected. Check the key in .env.") from None
    except anthropic.RateLimitError:
        raise LLMError("rate limited by the Anthropic API. Try again shortly.") from None
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error ({e.status_code}).") from None
    except anthropic.APIConnectionError:
        raise LLMError("could not reach the Anthropic API.") from None

    if response.stop_reason == "refusal":
        raise LLMError("the model declined this request.")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMError("the model returned no text.")
    return text


def complete_json(prompt: str, schema: dict) -> dict:
    """complete() with a schema, parsed into a dict."""
    text = complete(prompt, schema)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise LLMError("the model returned malformed JSON.") from None
    if not isinstance(parsed, dict):
        raise LLMError("the model returned JSON in an unexpected shape.")
    return parsed
