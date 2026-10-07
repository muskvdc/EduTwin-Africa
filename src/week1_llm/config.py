from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

def _streamlit_secret(name: str) -> Any:
    """Return one Streamlit secret without making Streamlit a hard dependency."""
    try:
        import streamlit as st

        secrets = st.secrets
        if hasattr(secrets, "get"):
            return secrets.get(name)
        return None
    except Exception:
        # Streamlit secrets are optional outside the Streamlit runtime.
        return None


def _secret_value(name: str) -> str | None:
    """Read a secret from the environment, then Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()

    streamlit_value = _streamlit_secret(name)
    if isinstance(streamlit_value, str) and streamlit_value.strip():
        return streamlit_value.strip()

    return None



def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    groq_api_key: str | None = _secret_value("GROQ_API_KEY")
    groq_base_url: str = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    local_embedding_dim: int = _int_env("LOCAL_EMBEDDING_DIM", 64)
    local_context_length: int = _int_env("LOCAL_CONTEXT_LENGTH", 32)
    local_hidden_dim: int = _int_env("LOCAL_HIDDEN_DIM", 128)
    context_token_budget: int = _int_env("LLM_CONTEXT_TOKEN_BUDGET", 8192)
    response_token_reserve: int = _int_env("LLM_RESPONSE_TOKEN_RESERVE", 1024)
    agent_max_iterations: int = _int_env("AGENT_MAX_ITERATIONS", 1)
    react_max_steps: int = _int_env("REACT_MAX_STEPS", 2)
    security_max_input_chars: int = _int_env("SECURITY_MAX_INPUT_CHARS", 12000)
    security_max_output_chars: int = _int_env("SECURITY_MAX_OUTPUT_CHARS", 24000)
    security_rate_limit_requests: int = _int_env("SECURITY_RATE_LIMIT_REQUESTS", 30)
    security_rate_limit_window_seconds: int = _int_env("SECURITY_RATE_LIMIT_WINDOW_SECONDS", 60)
    security_max_model_calls: int = _int_env("SECURITY_MAX_MODEL_CALLS", 10)
    security_max_tool_calls: int = _int_env("SECURITY_MAX_TOOL_CALLS", 4)
    security_max_estimated_tokens: int = _int_env("SECURITY_MAX_ESTIMATED_TOKENS", 30000)
    tavily_api_key: str | None = _secret_value("TAVILY_API_KEY")
    web_search_timeout_seconds: int = _int_env("WEB_SEARCH_TIMEOUT_SECONDS", 10)
    web_search_max_results: int = _int_env("WEB_SEARCH_MAX_RESULTS", 5)


config = Config()
