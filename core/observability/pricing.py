"""Turns token counts into money, from the prices a pack declares."""

from collections.abc import Mapping

from core.config import Price


class PriceBook:
    def __init__(self, prices: Mapping[str, Price]) -> None:
        self._prices = dict(prices)

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Dollar cost of one call, or ``None`` if the model's price is not known."""
        price = self._prices.get(model)
        return price.cost(input_tokens, output_tokens) if price else None
