"""A simple in-memory cache: remembers answers so the same question skips the AI."""

import time

MAX_SIZE = 100  # how many questions to remember
TTL_SECONDS = 3600  # forget an answer after 1 hour, in case the docs changed


class SimpleCache:
    """A dict in RAM. Everything is lost when the server restarts, and that's fine."""

    def __init__(self, max_size: int = MAX_SIZE, ttl_seconds: int = TTL_SECONDS):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.items = {}  # key -> {"value": ..., "saved_at": ...}

    def _key(self, question: str) -> str:
        """'  What is the SHIPPING  fee? ' -> 'what is the shipping fee?'"""
        return " ".join(question.lower().split())

    def get(self, question: str):
        """Return the saved answer, or None if it isn't saved or is too old."""
        key = self._key(question)
        item = self.items.get(key)
        if item is None:
            return None

        age = time.time() - item["saved_at"]
        if age > self.ttl_seconds:
            del self.items[key]  # too old, forget it
            return None

        return item["value"]

    def set(self, question: str, value) -> None:
        """Save an answer. If the cache is full, the oldest answer is removed."""
        key = self._key(question)

        # Remove the old copy first, so a re-saved question counts as the newest.
        self.items.pop(key, None)

        if len(self.items) >= self.max_size:
            oldest = next(iter(self.items))  # a dict keeps the order items went in
            del self.items[oldest]

        self.items[key] = {"value": value, "saved_at": time.time()}

    def clear(self) -> None:
        """Forget everything (used by tests)."""
        self.items = {}
