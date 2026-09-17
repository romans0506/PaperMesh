from unittest.mock import MagicMock, patch

import pytest
import requests

from papermesh.llm import MISSING_KEY_MESSAGE
from papermesh.summarizer import summarize_topic

PAPERS = [{"title": "Paper A", "abstract": "About A."}, {"title": "Paper B", "abstract": "About B."}]


@pytest.fixture(autouse=True)
def ollama_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUMMARY_BACKEND", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")


def test_ollama_summary_returns_model_text() -> None:
    response = MagicMock(ok=True, status_code=200)
    response.json.return_value = {"message": {"role": "assistant", "content": "  An overview.  "}}

    with patch("papermesh.llm.requests.post", return_value=response) as mock_post:
        summary = summarize_topic("topic", PAPERS)

    assert summary == "An overview."
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert "Paper A" in body["messages"][0]["content"]


def test_ollama_not_running_returns_message() -> None:
    with patch("papermesh.llm.requests.post", side_effect=requests.ConnectionError):
        summary = summarize_topic("topic", PAPERS)

    assert "Ollama isn't running" in summary


def test_claude_backend_without_key_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert summarize_topic("topic", PAPERS) == MISSING_KEY_MESSAGE
