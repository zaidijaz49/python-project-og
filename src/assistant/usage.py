"""Count the tokens Groq used for one chat and estimate what it would cost."""

import logging

logger = logging.getLogger("assistant")

# Groq's price in US dollars for 1 million tokens: (input, output).
# From https://console.groq.com/docs/models (checked 2026-10-23).
# Update these if Groq changes its prices or we add a model.
PRICES_PER_MILLION = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in US dollars, or None if we don't know this model's price."""
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class TokenUsage:
    """Adds up the tokens of every Groq call in one chat.

    One question can need several calls (ask -> run a tool -> answer),
    so we keep a running total.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0  # what we sent: prompt, message, tool results
        self.output_tokens = 0  # what the AI wrote back

    def add(self, usage) -> None:
        """Add one Groq response's `usage` (prompt_tokens, completion_tokens)."""
        self.calls += 1
        if usage is None:
            return
        self.input_tokens += usage.prompt_tokens or 0
        self.output_tokens += usage.completion_tokens or 0

    def log(self, model: str) -> None:
        """Write one line to the server log, e.g.
        Tokens: 1900 in, 60 out (2 calls) | model: gpt-oss-120b | cost: ~$0.000321
        """
        cost = estimate_cost(model, self.input_tokens, self.output_tokens)
        cost_text = "unknown" if cost is None else f"~${cost:.6f}"
        logger.info(
            "Tokens: %d in, %d out (%d calls) | model: %s | cost: %s",
            self.input_tokens,
            self.output_tokens,
            self.calls,
            model,
            cost_text,
        )
