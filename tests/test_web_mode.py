import importlib
import importlib.util
import sys
import types

# Keep route-dispatch tests runnable in a minimal test environment without
# installing the optional tokenizer dependency; production setup installs it.
if importlib.util.find_spec("tiktoken") is None:
    class _TestEncoding:
        def encode(self, text, disallowed_special=()):
            return text.split()

    _fake_tiktoken = types.ModuleType("tiktoken")
    _fake_tiktoken.encoding_for_model = lambda model: _TestEncoding()
    sys.modules["tiktoken"] = _fake_tiktoken

web_module = importlib.import_module("web.app")


class FakePipeline:
    def chat(self, message, temperature):
        return f"normal:{message}:{temperature}"

    def multi_agent_chat(self, message, temperature):
        return {
            "answer": f"multi:{message}:{temperature}",
            "trace": [{"agent": "Orchestrator", "status": "complete", "summary": "Done"}],
            "iterations": 1,
            "review_status": "approved",
        }


def test_chat_request_defaults_to_normal_mode():
    request = web_module.ChatRequest(message="Hello")
    assert request.mode == "normal"


def test_api_dispatches_normal_and_multi_agent_modes(monkeypatch):
    monkeypatch.setattr(web_module, "pipeline", FakePipeline())
    normal = web_module.chat(web_module.ChatRequest(message="Hello"))
    multi = web_module.chat(
        web_module.ChatRequest(message="Hello", mode="multi_agent")
    )
    assert normal["mode"] == "normal"
    assert normal["answer"].startswith("normal:Hello")
    assert multi["mode"] == "multi_agent"
    assert multi["review_status"] == "approved"
    assert multi["trace"][0]["agent"] == "Orchestrator"
