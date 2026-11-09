"""Unit tests for recorder/pricing.py (M2-T3). Pure function, no mocking, no live calls."""

import pytest

from recorder.pricing import estimate_cost_inr, load_price_table, PRICING_TABLE_PATH


@pytest.fixture
def price_table():
    return {
        "as_of": "2026-11-09",
        "currency": "INR",
        "models": {
            "groq/llama-3.3-70b-versatile": {"inr_per_1k_in": 0.04, "inr_per_1k_out": 0.08},
        },
    }


def test_estimate_cost_known_model(price_table):
    cost = estimate_cost_inr(
        "groq/llama-3.3-70b-versatile", tokens_in=1000, tokens_out=500, price_table=price_table
    )
    assert cost == pytest.approx(0.04 + 0.04)


def test_estimate_cost_zero_tokens(price_table):
    cost = estimate_cost_inr(
        "groq/llama-3.3-70b-versatile", tokens_in=0, tokens_out=0, price_table=price_table
    )
    assert cost == 0


def test_unknown_model_raises(price_table):
    with pytest.raises(ValueError):
        estimate_cost_inr("unknown/model", tokens_in=100, tokens_out=100, price_table=price_table)


def test_pricing_table_file_loads_and_has_as_of():
    table = load_price_table(PRICING_TABLE_PATH)
    assert "as_of" in table
    assert "models" in table
    assert len(table["models"]) > 0
