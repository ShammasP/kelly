import json
import types
import pytest

import kelly_groq


class DummyResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def test_respond_groq_falls_back_without_key(monkeypatch, tmp_path):
    # Ensure no GROQ_API_KEY in env
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = kelly_groq.respond_groq("test")
    assert isinstance(out, str) and out.startswith("[LLM error:")


def test_respond_groq_parses_choice(monkeypatch):
    # Provide a fake API key
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # mocked response should be returned directly in LLM-only mode
    mocked_content = "This is a mocked poem."
    body = {"choices": [{"message": {"content": mocked_content}}]}

    def fake_post(url, headers=None, data=None, timeout=None):
        return DummyResponse(200, body)

    monkeypatch.setattr(kelly_groq.requests, "post", fake_post)
    # Ensure tests use the HTTP fallback even if the SDK is installed in the env
    monkeypatch.setattr(kelly_groq, "_HAVE_GROQ_SDK", False)

    out = kelly_groq.respond_groq("Will AI replace scientists?")
    assert out == mocked_content
