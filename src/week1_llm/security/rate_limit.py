from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int = 0


class RateLimiter:
    """Thread-safe sliding-window rate limiter for a single process.

    This is appropriate for the local educational project. A multi-process or
    distributed deployment should move the counter to shared infrastructure.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            queue = self._requests[key]
            cutoff = now - self.window_seconds
            while queue and queue[0] <= cutoff:
                queue.popleft()

            if len(queue) >= self.max_requests:
                retry = max(1, int(self.window_seconds - (now - queue[0])))
                return RateLimitDecision(False, 0, retry)

            queue.append(now)
            return RateLimitDecision(True, self.max_requests - len(queue), 0)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._requests.clear()
            else:
                self._requests.pop(key, None)
