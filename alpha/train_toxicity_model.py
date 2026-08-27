"""CLI to train the order-flow toxicity classifier from real ingested
history in DuckDB.

NOT run as part of this environment's current deliverable: see the module
docstring in alpha/toxicity_model.py -- a real toxicity label needs several
sequential snapshots per contract, and this session only has a single
fixture snapshot ingested (see docs/WHITEPAPER.md rollout notes). This
script is fully functional and will train + save a real model once
`python -m data.ingest --mode live --max-polls 0` (or a longer fixture-mode
polling run) has accumulated enough sequential history in DuckDB; running
it too early fails loudly with a clear "not enough history" message rather
than silently training garbage.

Label construction
-------------------
A quote at time t is labeled "toxic" (y=1) if the mid price moves against
a passive quote posted at the touch by more than the then-quoted spread
within the next `horizon` snapshots -- i.e. a market maker resting at the
bid/ask would have been adversely selected by more than it was being paid
to take on that risk. This is a standard proxy for adverse-selection risk
in the market-making toxicity literature (informed-flow detection in the
spirit of Easley/O'Hara-style VPIN work), not a sophisticated learned
label -- it's a reasonable default to get a real pipeline running; swap in
a better-validated label as real data accumulates.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from alpha.features import FEATURE_COLUMNS, compute_features
from alpha.toxicity_model import ToxicityClassifier
from data.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

MIN_DISTINCT_TIMESTAMPS = 5


def build_labels(features_df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Adds a `toxic` (0/1) column via the forward-looking adverse-selection
    proxy described in the module docstring. Drops rows with no forward
    observation (the last `horizon` snapshots of each contract's history).
    """
    df = features_df.sort_values(["expiry", "strike", "option_type", "ts"]).copy()
    grp = df.groupby(["expiry", "strike", "option_type"], group_keys=False)
    df["mid_fwd"] = grp["mid"].shift(-horizon)
    fwd_move = (df["mid_fwd"] - df["mid"]).abs()
    df["toxic"] = (fwd_move > df["spread"]).astype(int)
    return df.dropna(subset=["mid_fwd"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toxicity classifier from ingested DuckDB history")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--db", default="data/db/quant.duckdb")
    parser.add_argument("--horizon", type=int, default=3, help="Snapshots ahead used for the toxicity label")
    parser.add_argument("--out", default=None, help="Output path; defaults to alpha/models/toxicity_<symbol>.joblib")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with DuckDBStore(args.db) as store:
        timestamps = store.distinct_timestamps(args.symbol)
        required = MIN_DISTINCT_TIMESTAMPS + args.horizon
        if len(timestamps) < required:
            logger.error(
                "Not enough sequential history for %s to train a toxicity model: %d distinct "
                "snapshot timestamps in %s, need at least %d (horizon=%d snapshots-ahead label). "
                "Run `python -m data.ingest --mode live --max-polls 0` for longer to accumulate "
                "sequential history first.",
                args.symbol,
                len(timestamps),
                args.db,
                required,
                args.horizon,
            )
            sys.exit(1)
        raw = store.query_range(args.symbol)

    features = compute_features(raw)
    labeled = build_labels(features, horizon=args.horizon)

    X = labeled[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    y = labeled["toxic"].to_numpy()

    if len(set(y.tolist())) < 2:
        logger.error(
            "Labeled data contains only one class (%d samples, all toxic=%s) -- cannot train a "
            "classifier. Need more varied market history.",
            len(y),
            y[0] if len(y) else "?",
        )
        sys.exit(1)

    model = ToxicityClassifier().train(X, y)

    out_path = Path(args.out or f"alpha/models/toxicity_{args.symbol.lower()}.joblib")
    model.save(out_path)

    print(f"Trained toxicity model on {len(X)} samples ({int(y.sum())} labeled toxic). Saved to {out_path}")
    print("Feature importances:", model.feature_importances())


if __name__ == "__main__":
    main()
