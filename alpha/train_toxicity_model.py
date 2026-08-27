"""CLI to train the order-flow toxicity classifier from real ingested
history in DuckDB.

Works against any ingestion source with enough sequential history --
`--mode bhavcopy`/`bhavcopy-local` real EOD data (§3 "Phase 1.5" of
docs/WHITEPAPER.md) or, once available, `--mode live` intraday quotes.
Fails loudly with an actionable message, rather than silently training on
too little or degenerate data, in two cases: fewer than
`MIN_DISTINCT_TIMESTAMPS + horizon` snapshots ingested, or a labeled
dataset that ends up single-class.

Label construction
-------------------
A quote at time t is labeled "toxic" (y=1) if the mid price moves against
a passive quote posted at the touch by more than the "was this move big
enough to matter" threshold within the next `horizon` snapshots -- i.e. a
market maker resting at the bid/ask would have been adversely selected by
more than it was being paid to take on that risk. This is a standard proxy
for adverse-selection risk in the market-making toxicity literature
(informed-flow detection in the spirit of Easley/O'Hara-style VPIN work),
not a sophisticated learned label -- it's a reasonable default to get a
real pipeline running; swap in a better-validated label as real data
accumulates.

The threshold is the then-quoted spread when one is available (intraday
quote data), or `MIN_RELATIVE_MOVE_THRESHOLD` (a percentage of mid) when
it isn't -- EOD-only sources like Bhavcopy report bid=ask=0 for every row,
so spread alone would degenerate into "labeled toxic whenever the price
moved at all," which swamps the label with noise rather than signal (see
`alpha/features.py::compute_features` for the matching `mid` fallback that
makes this threshold computable in the first place).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.features import FEATURE_COLUMNS, compute_features
from alpha.toxicity_model import ToxicityClassifier
from data.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

MIN_DISTINCT_TIMESTAMPS = 5

# Fallback "did this move enough to matter" threshold, as a fraction of
# mid, used only when `spread` is zero for a row. `spread` is the natural
# threshold for intraday quote data (was the forward move bigger than what
# a resting quote was being paid to take on?), but EOD-only sources (e.g.
# NSE Bhavcopy, see docs/WHITEPAPER.md) report bid=ask=0 for every row, so
# spread is always 0 there. Using it directly would label literally any
# nonzero day-to-day price change as "toxic" -- not a meaningful
# adverse-selection signal, just noise. This is a reasonable default, not
# a validated-on-real-data choice -- see this module's docstring.
MIN_RELATIVE_MOVE_THRESHOLD = 0.01


def build_labels(features_df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Adds a `toxic` (0/1) column via the forward-looking adverse-selection
    proxy described in the module docstring. Drops rows with no forward
    observation (the last `horizon` snapshots of each contract's history).
    """
    df = features_df.sort_values(["expiry", "strike", "option_type", "ts"]).copy()
    grp = df.groupby(["expiry", "strike", "option_type"], group_keys=False)
    df["mid_fwd"] = grp["mid"].shift(-horizon)
    fwd_move = (df["mid_fwd"] - df["mid"]).abs()
    effective_threshold = np.where(
        df["spread"] > 0, df["spread"], MIN_RELATIVE_MOVE_THRESHOLD * df["mid"].abs()
    )
    df["toxic"] = (fwd_move > effective_threshold).astype(int)
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
