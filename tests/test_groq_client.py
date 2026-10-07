import requests
import pytest

from week1_llm.api.groq_client import GroqClient


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "hello"}}]}


def test_groq_client_sends_completion_reserve(monkeypatch):
    captured = {}
    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    client = GroqClient(api_key="test-key")
    answer = client.chat(
        [{"role": "user", "content": "hello"}],
        max_completion_tokens=512,
    )
    assert answer == "hello"
    assert captured["payload"]["max_completion_tokens"] == 512
    assert "max_tokens" not in captured["payload"]


def test_groq_client_rejects_invalid_completion_reserve():
    client = GroqClient(api_key="test-key")
    with pytest.raises(ValueError, match="max_completion_tokens"):
        client.chat([{"role": "user", "content": "hello"}], max_completion_tokens=0)
