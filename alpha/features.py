"""Microstructure feature engineering shared by the toxicity classifier
(alpha/toxicity_model.py) and the quoting engine.

Operates on a pandas DataFrame in the shape returned by
`data.duckdb_store.DuckDBStore.query_range` (one row per
timestamp/expiry/strike/option_type quote), computing, per contract, across
consecutive snapshots in time:

  - order_flow_imbalance (OFI): (bid_qty - ask_qty) / (bid_qty + ask_qty),
    in [-1, 1]. Positive -> more resting size on the bid (buy pressure);
    negative -> more resting size on the ask (sell pressure).
  - delta_oi: change in open interest since the previous snapshot (falls
    back to the exchange-reported `change_in_oi` when this is the first
    snapshot seen for a contract in the queried window).
  - spread_velocity: rate of change of the quoted spread (ask - bid) per
    second between consecutive snapshots -- a fast-widening spread is a
    classic precursor to informed flow / adverse selection.
  - volume_imbalance: change in cumulative traded volume since the previous
    snapshot (a burst of volume with no corresponding OI change often
    signals aggressive, potentially informed, order flow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["ofi", "delta_oi", "spread_velocity", "volume_imbalance"]

_GROUP_KEYS = ["expiry", "strike", "option_type"]


def order_flow_imbalance(bid_qty: np.ndarray, ask_qty: np.ndarray) -> np.ndarray:
    denom = bid_qty + ask_qty
    return np.where(denom > 0, (bid_qty - ask_qty) / np.maximum(denom, 1), 0.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of `df` with `mid`, `spread`, and `FEATURE_COLUMNS`
    added. Rows are grouped by (expiry, strike, option_type) and features
    are computed against each contract's own immediately preceding
    snapshot -- the first observation of a contract in the window has no
    predecessor, so its delta-based features are 0 (delta_oi falls back to
    the exchange-reported change_in_oi instead).

    `df` must include an `ltp` column (present on every row returned by
    `data.duckdb_store.DuckDBStore.query_range`, per `data/schema.sql`'s
    `NOT NULL` constraint). `mid` mirrors `core.option_chain.OptionContract
    .mid`'s fallback exactly: `(bid + ask) / 2` when both are quoted,
    else `ltp`. This matters beyond cosmetics -- EOD-only sources (e.g.
    NSE Bhavcopy, see docs/WHITEPAPER.md) report `bid = ask = 0` for every
    row; without the fallback, `mid` would be a constant `0.0` for that
    entire data source, silently making every mid-price-based signal
    (including `alpha/train_toxicity_model.py`'s forward-looking label)
    degenerate rather than merely low-signal.
    """
    out = df.sort_values([*_GROUP_KEYS, "ts"]).reset_index(drop=True).copy()
    grp = out.groupby(_GROUP_KEYS, group_keys=False)

    has_quote = (out["bid"] > 0) & (out["ask"] > 0)
    out["mid"] = np.where(has_quote, (out["bid"] + out["ask"]) / 2.0, out["ltp"])
    out["spread"] = out["ask"] - out["bid"]
    out["ofi"] = order_flow_imbalance(out["bid_qty"].to_numpy(dtype=float), out["ask_qty"].to_numpy(dtype=float))

    prev_ts = grp["ts"].shift(1)
    dt_sec = (out["ts"] - prev_ts).dt.total_seconds()

    prev_spread = grp["spread"].shift(1)
    out["spread_velocity"] = ((out["spread"] - prev_spread) / dt_sec.replace(0, np.nan)).fillna(0.0)

    prev_volume = grp["volume"].shift(1)
    out["volume_imbalance"] = (out["volume"] - prev_volume).fillna(0.0)

    prev_oi = grp["oi"].shift(1)
    delta_oi = out["oi"] - prev_oi
    out["delta_oi"] = delta_oi.fillna(out["change_in_oi"])

    return out
