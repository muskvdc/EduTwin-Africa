import requests

from week1_llm.api.groq_client import GroqClient, GroqRateLimitError


class FakeResponse:
    def __init__(self, status_code, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


def test_groq_client_retries_short_429(monkeypatch):
    responses = [
        FakeResponse(429, {"retry-after": "0.01"}),
        FakeResponse(200, payload={"choices": [{"message": {"content": "ok"}}]}),
    ]
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr("week1_llm.api.groq_client.requests.post", fake_post)
    client = GroqClient(api_key="test")
    client.max_429_retries = 1
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(calls) == 2


def test_groq_client_surfaces_long_429_without_waiting(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(429, {"retry-after": "2m"})

    monkeypatch.setattr("week1_llm.api.groq_client.requests.post", fake_post)
    client = GroqClient(api_key="test")
    client.max_429_retries = 1
    error = None
    try:
        client.chat([{"role": "user", "content": "hi"}])
    except GroqRateLimitError as exc:
        error = exc
    assert error is not None
    assert error.wait_seconds == 120
    assert "Normal Chat" in error.user_message
