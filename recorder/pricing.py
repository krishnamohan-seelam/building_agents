"""Estimates ₹ cost per LLM call from a price table (M2-T3)."""

import json
from pathlib import Path

PRICING_TABLE_PATH = Path(__file__).resolve().parent / "pricing_table.json"


def load_price_table(path: Path = PRICING_TABLE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def estimate_cost_inr(
    model: str, tokens_in: int, tokens_out: int, price_table: dict
) -> float:
    """Looks up `model` in price_table["models"]; raises if the model is unknown."""
    models = price_table.get("models", {})
    if model not in models:
        raise ValueError(f"No pricing entry for model {model!r} in price table")

    rates = models[model]
    cost = (tokens_in / 1000) * rates["inr_per_1k_in"] + (tokens_out / 1000) * rates["inr_per_1k_out"]
    return round(cost, 4)
