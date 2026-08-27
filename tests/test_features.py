"""Tests for alpha/features.py's `mid` calculation, in particular the
ltp fallback for EOD-only data sources (bid=ask=0, e.g. NSE Bhavcopy) --
see docs/WHITEPAPER.md and alpha/features.py's module/function docstrings
for why this matters (it silently broke toxicity-label construction
before the fix this test guards against regressing).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from alpha.features import compute_features


def _row(ts, strike, bid, ask, ltp, oi=1000, change_in_oi=0, volume=100, bid_qty=0, ask_qty=0):
    return {
        "ts": ts,
        "expiry": date(2026, 9, 3),
        "strike": strike,
        "option_type": "CE",
        "bid": bid,
        "bid_qty": bid_qty,
        "ask": ask,
        "ask_qty": ask_qty,
        "oi": oi,
        "change_in_oi": change_in_oi,
        "volume": volume,
        "ltp": ltp,
    }


def test_mid_uses_book_when_quoted():
    df = pd.DataFrame(
        [_row(datetime(2026, 8, 1), 25000.0, bid=100.0, ask=102.0, ltp=101.5, bid_qty=50, ask_qty=60)]
    )
    features = compute_features(df)
    assert features["mid"].iloc[0] == 101.0  # (100+102)/2, not ltp
    assert features["spread"].iloc[0] == 2.0


def test_mid_falls_back_to_ltp_when_no_quote():
    df = pd.DataFrame([_row(datetime(2026, 8, 1), 25000.0, bid=0.0, ask=0.0, ltp=87.35)])
    features = compute_features(df)
    assert features["mid"].iloc[0] == 87.35
    assert features["spread"].iloc[0] == 0.0


def test_mid_falls_back_when_only_one_side_quoted():
    # A zero on just one side (e.g. no resting bid on a deep-OTM strike)
    # should also fall back -- matches core.option_chain.OptionContract.mid,
    # which requires *both* bid and ask to be positive.
    df = pd.DataFrame([_row(datetime(2026, 8, 1), 25000.0, bid=0.0, ask=5.0, ltp=2.5)])
    features = compute_features(df)
    assert features["mid"].iloc[0] == 2.5


def test_mid_series_across_eod_history_tracks_real_price_moves():
    """Regression test for the original bug: mid was a constant 0.0 across
    an entire EOD history because it never fell back to ltp, which made
    every mid-price-based signal (including the toxicity label) degenerate
    rather than merely low-signal.
    """
    rows = [
        _row(datetime(2026, 7, 22), 25000.0, bid=0.0, ask=0.0, ltp=120.0),
        _row(datetime(2026, 7, 23), 25000.0, bid=0.0, ask=0.0, ltp=135.0),
        _row(datetime(2026, 7, 24), 25000.0, bid=0.0, ask=0.0, ltp=110.0),
    ]
    df = pd.DataFrame(rows)
    features = compute_features(df).sort_values("ts")
    assert list(features["mid"]) == [120.0, 135.0, 110.0]
    assert not (features["mid"] == 0.0).any()
