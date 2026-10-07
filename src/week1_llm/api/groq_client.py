from __future__ import annotations

import os
import random
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class GroqRateLimitError(RuntimeError):
    """Raised when Groq returns a 429 after safe retry handling."""

    def __init__(self, message: str, *, wait_seconds: float | None = None):
        super().__init__(message)
        self.wait_seconds = wait_seconds
        if wait_seconds is not None and wait_seconds > 0:
            if wait_seconds < 60:
                wait_text = f"about {max(1, round(wait_seconds))} seconds"
            else:
                wait_text = f"about {round(wait_seconds / 60)} minutes"
            self.user_message = (
                "Please wait "
                + wait_text
                + " and try again, or use Normal Chat for a lighter request."
            )
        else:
            self.user_message = (
                "Please wait a moment and try again, or use Normal Chat for a lighter request."
            )


class GroqClient:
    """
    Small OpenAI-compatible HTTP wrapper matching the Week 1 API lesson.

    Includes conservative 429 handling because Multi-Agent mode can make
    several model calls in one learner request. Groq documents RPM/TPM and
    rate-limit reset headers; this client respects those signals instead of
    immediately surfacing a raw HTTPError.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.default_model = default_model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )
        self.max_429_retries = max(
            0, int(os.getenv("GROQ_MAX_429_RETRIES", "1"))
        )
        self.max_429_wait_seconds = max(
            1.0, float(os.getenv("GROQ_MAX_429_WAIT_SECONDS", "15"))
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _parse_duration(value: str | None) -> float | None:
        if not value:
            return None
        text = str(value).strip().lower()
        try:
            return max(0.0, float(text))
        except ValueError:
            pass

        total = 0.0
        matched = False
        for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
            matched = True
            number = float(amount)
            if unit == "h":
                total += number * 3600
            elif unit == "m":
                total += number * 60
            else:
                total += number
        return total if matched else None

    def _rate_limit_wait(self, response: requests.Response) -> float | None:
        # Prefer the server's explicit retry-after instruction.
        wait = self._parse_duration(response.headers.get("retry-after"))
        if wait is not None:
            return wait

        # Groq exposes token-window reset information even when retry-after
        # is absent. This is especially useful for Multi-Agent's several
        # sequential calls.
        return self._parse_duration(
            response.headers.get("x-ratelimit-reset-tokens")
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        timeout: int = 60,
        max_completion_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Put it in your .env file."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            if max_completion_tokens < 1:
                raise ValueError("max_completion_tokens must be positive")
            payload["max_completion_tokens"] = max_completion_tokens

        attempts = 0
        while True:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected API response: {data}"
                    ) from exc

            wait = self._rate_limit_wait(response)
            attempts += 1

            # Never sleep for a long daily-limit reset. Give the learner a
            # useful message instead. Short TPM/RPM resets are safe to retry.
            if (
                attempts > self.max_429_retries
                or wait is None
                or wait > self.max_429_wait_seconds
            ):
                raise GroqRateLimitError(
                    "Groq returned HTTP 429 Too Many Requests.",
                    wait_seconds=wait,
                )

            # A small jitter avoids several rapid requests waking together.
            time.sleep(max(0.5, wait) + random.uniform(0.0, 0.35))
