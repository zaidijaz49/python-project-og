"""Tests for the in-memory cache in cache.py."""

from assistant.cache import SimpleCache


def test_missing_question_returns_none():
    cache = SimpleCache()
    assert cache.get("What is the shipping fee?") is None


def test_saved_answer_is_returned():
    cache = SimpleCache()
    cache.set("What is the shipping fee?", "Rs. 200")
    assert cache.get("What is the shipping fee?") == "Rs. 200"


def test_case_and_spaces_do_not_matter():
    cache = SimpleCache()
    cache.set("What is the shipping fee?", "Rs. 200")
    assert cache.get("  what IS the   SHIPPING fee?  ") == "Rs. 200"


def test_different_question_is_not_found():
    cache = SimpleCache()
    cache.set("What is the shipping fee?", "Rs. 200")
    assert cache.get("How long does a refund take?") is None


def test_old_answer_expires(monkeypatch):
    cache = SimpleCache(ttl_seconds=60)

    # Pretend the clock is at 1000 seconds when we save.
    monkeypatch.setattr("assistant.cache.time.time", lambda: 1000)
    cache.set("hi", "Hello!")

    # 30 seconds later: still fresh.
    monkeypatch.setattr("assistant.cache.time.time", lambda: 1030)
    assert cache.get("hi") == "Hello!"

    # 61 seconds later: too old.
    monkeypatch.setattr("assistant.cache.time.time", lambda: 1061)
    assert cache.get("hi") is None


def test_oldest_answer_is_removed_when_full():
    cache = SimpleCache(max_size=2)
    cache.set("first", 1)
    cache.set("second", 2)
    cache.set("third", 3)  # cache is full, so "first" goes

    assert cache.get("first") is None
    assert cache.get("second") == 2
    assert cache.get("third") == 3


def test_clear_forgets_everything():
    cache = SimpleCache()
    cache.set("hi", "Hello!")
    cache.clear()
    assert cache.get("hi") is None
