"""Tests for the forward-looking adverse-selection label construction in
alpha/train_toxicity_model.py, and the minimum-history guard in its CLI.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from alpha.features import compute_features
from alpha.train_toxicity_model import build_labels
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
