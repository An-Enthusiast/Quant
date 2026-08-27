"""Tests for the Avellaneda-Stoikov reservation price / spread and the
quote-engine's toxicity-driven widening.
"""

from __future__ import annotations

import pytest

from alpha.avellaneda_stoikov import compute_quote, optimal_spread, reservation_price
from alpha.quote_engine import QuoteParams, quote_contract


def test_reservation_price_symmetric_at_zero_inventory():
    r = reservation_price(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.15, time_to_close=0.02)
    assert r == pytest.approx(100.0)


def test_reservation_price_skews_down_for_long_inventory():
    r = reservation_price(mid=100.0, inventory=10.0, gamma=0.1, sigma=0.15, time_to_close=0.02)
    assert r < 100.0


def test_reservation_price_skews_up_for_short_inventory():
    r = reservation_price(mid=100.0, inventory=-10.0, gamma=0.1, sigma=0.15, time_to_close=0.02)
    assert r > 100.0


def test_spread_widens_with_time_to_close():
    s1 = optimal_spread(gamma=0.1, sigma=0.15, time_to_close=0.01, kappa=1.5)
    s2 = optimal_spread(gamma=0.1, sigma=0.15, time_to_close=0.1, kappa=1.5)
    assert s2 > s1


def test_spread_widens_with_volatility():
    s1 = optimal_spread(gamma=0.1, sigma=0.10, time_to_close=0.02, kappa=1.5)
    s2 = optimal_spread(gamma=0.1, sigma=0.30, time_to_close=0.02, kappa=1.5)
    assert s2 > s1


def test_spread_requires_positive_gamma_and_kappa():
    with pytest.raises(ValueError):
        optimal_spread(gamma=0.0, sigma=0.15, time_to_close=0.02, kappa=1.5)
    with pytest.raises(ValueError):
        optimal_spread(gamma=0.1, sigma=0.15, time_to_close=0.02, kappa=0.0)


def test_compute_quote_bid_below_ask():
    q = compute_quote(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.15, time_to_close=0.02, kappa=1.5)
    assert q.bid < q.reservation_price < q.ask


def test_toxicity_score_widens_spread():
    key = ("NIFTY", None, 25000.0, None)
    base = quote_contract(key, mid=100.0, inventory=0.0, sigma=0.15, time_to_close=0.02, toxicity_score=0.0)
    widened = quote_contract(key, mid=100.0, inventory=0.0, sigma=0.15, time_to_close=0.02, toxicity_score=1.0)
    assert widened.spread > base.spread


def test_toxicity_widening_is_capped():
    key = ("NIFTY", None, 25000.0, None)
    params = QuoteParams(toxicity_beta=100.0, max_toxicity_widen=3.0)
    base = quote_contract(key, mid=100.0, inventory=0.0, sigma=0.15, time_to_close=0.02, params=params, toxicity_score=0.0)
    widened = quote_contract(key, mid=100.0, inventory=0.0, sigma=0.15, time_to_close=0.02, params=params, toxicity_score=1.0)
    assert widened.spread == pytest.approx(base.spread * params.max_toxicity_widen, rel=1e-6)
