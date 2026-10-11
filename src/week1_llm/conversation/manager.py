from __future__ import annotations



class ConversationManager:
    """Keeps a bounded multi-turn message history."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def clear(self) -> None:
        self.messages.clear()

    def restore(self, messages: list[dict[str, str]]) -> None:
        """Restore a saved conversation into the bounded message history."""
        self.messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
            if message.get("role") in ("user", "assistant")
        ]
        self._trim()

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def as_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

