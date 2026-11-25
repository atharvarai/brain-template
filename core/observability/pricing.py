"""Turns token counts into money, from the prices a pack declares."""

from collections.abc import Mapping

from core.config import Price


class PriceBook:
    def __init__(self, prices: Mapping[str, Price]) -> None:
        self._prices = dict(prices)

    def cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Dollar cost of one call, or ``None`` if the model's price is not known."""
        price = self._prices.get(model)
        if price is None:
            return None
        return price.cost(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
