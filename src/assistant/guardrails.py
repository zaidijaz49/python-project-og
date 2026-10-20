"""Input guard: clean user text and block tricky input before it reaches the AI."""

import unicodedata

# Phrases people use to try to override the AI's instructions (lowercase).
BLOCKED_PHRASES = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard previous instructions",
    "disregard all previous instructions",
    "forget your instructions",
    "forget all previous instructions",
    "system prompt",
    "you are now",
    "pretend you are",
    "<user_text>",  # trying to break out of our tags
    "</user_text>",
]

MAX_SAME_CHAR = 50  # the same character 50 times in a row is not normal text


class GuardrailError(Exception):
    """User text was blocked by the guard."""


def clean_text(text: str) -> str:
    """Make text safe to check: normal characters, no hidden ones, no edge spaces."""
    # 1. Turn look-alike characters into normal ones.
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove hidden characters (zero-width etc.), but keep new lines and tabs.
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

    # 3. Remove spaces at the start and end.
    return text.strip()


def check_text(text: str) -> str:
    """Clean the text, block it if it looks bad, else return the clean text."""
    text = clean_text(text)

    # Check 1: is anything left after cleaning?
    if not text:
        raise GuardrailError("Text is empty.")

    # Check 2: tricky phrases. Lowercase + single spaces so "IGNORE   all" matches.
    simple = " ".join(text.lower().split())
    for phrase in BLOCKED_PHRASES:
        if phrase in simple:
            raise GuardrailError(
                "Input blocked: looks like an attempt to override instructions."
            )

    # Check 3: the same character repeated too many times.
    for ch in set(text):
        if ch * MAX_SAME_CHAR in text:
            raise GuardrailError("Input blocked: too much repetition.")

    return text
