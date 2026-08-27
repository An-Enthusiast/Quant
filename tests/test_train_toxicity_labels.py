"""Tests for the forward-looking adverse-selection label construction in
alpha/train_toxicity_model.py, and the minimum-history guard in its CLI.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from alpha.features import compute_features
from alpha.train_toxicity_model import MIN_RELATIVE_MOVE_THRESHOLD, build_labels
from backtest.synthetic_ticks import generate_tick_series
from data.duckdb_store import DuckDBStore


def _features_df(n_ticks=15):
    expiry = date.today().fromordinal(date.today().toordinal() + 7)
    ticks = generate_tick_series(
        "NIFTY", spot0=25000.0, strike_step=50.0, base_iv=0.13, expiry=expiry, n_ticks=n_ticks, tick_interval_sec=5.0
    )
    rows = []
    for snap in ticks:
        for c in snap.contracts:
            rows.append(
                {
                    "ts": snap.timestamp,
                    "expiry": c.expiry,
                    "strike": c.strike,
                    "option_type": c.option_type.value,
                    "bid": c.bid,
                    "bid_qty": c.bid_qty,
                    "ask": c.ask,
                    "ask_qty": c.ask_qty,
                    "oi": c.oi,
                    "change_in_oi": c.change_in_oi,
                    "volume": c.volume,
                    "ltp": c.ltp,
                }
            )
    import pandas as pd

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return compute_features(df)


def test_build_labels_produces_binary_toxic_column():
    features = _features_df()
    labeled = build_labels(features, horizon=3)
    assert set(labeled["toxic"].unique()) <= {0, 1}
    assert "mid_fwd" in labeled.columns
    assert len(labeled) < len(features)  # last `horizon` rows per contract are dropped


def test_build_labels_drops_rows_without_forward_observation():
    features = _features_df(n_ticks=5)
    labeled = build_labels(features, horizon=3)
    # every remaining row must have had a valid forward-looking mid
    assert labeled["mid_fwd"].notna().all()


def _eod_row(ts, strike, ltp):
    """A Bhavcopy-shaped row: no real bid/ask (see data/bhavcopy_loader.py)."""
    return {
        "ts": ts,
        "expiry": date(2026, 9, 3),
        "strike": strike,
        "option_type": "CE",
        "bid": 0.0,
        "bid_qty": 0,
        "ask": 0.0,
        "ask_qty": 0,
        "oi": 1000,
        "change_in_oi": 0,
        "volume": 100,
        "ltp": ltp,
    }


def test_build_labels_uses_relative_threshold_when_spread_is_zero():
    """EOD-only data (spread always 0, per data/bhavcopy_loader.py) must
    not degenerate into "toxic whenever the price moves at all" -- a small
    move stays non-toxic, a move past MIN_RELATIVE_MOVE_THRESHOLD is toxic.
    """
    day1 = datetime(2026, 7, 22)
    day2 = day1 + timedelta(days=1)
    base_ltp = 100.0
    small_move = base_ltp * MIN_RELATIVE_MOVE_THRESHOLD * 0.1  # well under threshold
    large_move = base_ltp * MIN_RELATIVE_MOVE_THRESHOLD * 10  # well over threshold
    rows = [
        _eod_row(day1, 25000.0, ltp=base_ltp),
        _eod_row(day2, 25000.0, ltp=base_ltp + small_move),
        _eod_row(day1, 25100.0, ltp=base_ltp),
        _eod_row(day2, 25100.0, ltp=base_ltp + large_move),
    ]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    features = compute_features(df)
    assert (features["spread"] == 0.0).all()  # sanity: this is the EOD case

    labeled = build_labels(features, horizon=1)
    by_strike = labeled.set_index("strike")["toxic"]
    assert by_strike.loc[25000.0] == 0
    assert by_strike.loc[25100.0] == 1


def test_build_labels_eod_threshold_is_not_all_one_class_over_real_month(tmp_path):
    """Guards against the exact regression found on the real 1-month
    Bhavcopy archive: before the mid/threshold fixes, every row labeled
    toxic=0 (mid was a constant 0.0), which fails ToxicityClassifier's
    "need both classes" check outright. This doesn't assert a specific
    class balance (that depends on real market moves), only that the
    label isn't trivially degenerate on a dataset with real day-to-day
    price variation.
    """
    rng_prices = [100.0, 100.3, 99.0, 104.0, 96.0, 110.0, 105.0]  # one small (<1%) move, several large ones
    base_day = datetime(2026, 7, 22)
    rows = [_eod_row(base_day + timedelta(days=i), 25000.0, ltp=p) for i, p in enumerate(rng_prices)]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    labeled = build_labels(compute_features(df), horizon=1)
    assert set(labeled["toxic"].unique()) == {0, 1}


def test_cli_exits_cleanly_on_insufficient_history(tmp_path):
    db_path = tmp_path / "tiny.duckdb"
    with DuckDBStore(db_path) as store:
        expiry = date.today().fromordinal(date.today().toordinal() + 7)
        ticks = generate_tick_series(
            "NIFTY", spot0=25000.0, strike_step=50.0, base_iv=0.13, expiry=expiry, n_ticks=2, tick_interval_sec=5.0
        )
        for snap in ticks:
            store.insert_snapshot(snap)

    result = subprocess.run(
        [sys.executable, "-m", "alpha.train_toxicity_model", "--symbol", "NIFTY", "--db", str(db_path)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0
