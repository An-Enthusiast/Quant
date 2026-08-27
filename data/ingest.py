"""CLI orchestrator for populating DuckDB with option-chain snapshots.

Examples
--------
Fixture-mode one-shot ingest (the current default -- no live NSE calls,
see docs/WHITEPAPER.md):

    python -m data.ingest --mode fixture --symbols NIFTY,BANKNIFTY

Live continuous polling (once network access to nseindia.com is available):

    python -m data.ingest --mode live --symbols NIFTY,BANKNIFTY --interval 5 --max-polls 0
"""

from __future__ import annotations

import argparse
import logging

from data.duckdb_store import DuckDBStore
from data.nsepython_poller import poll_forever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NSE option-chain snapshots into DuckDB")
    parser.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    parser.add_argument("--symbols", default="NIFTY,BANKNIFTY", help="Comma-separated underlyings")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between polls")
    parser.add_argument(
        "--max-polls", type=int, default=1, help="Number of poll rounds; 0 means run forever"
    )
    parser.add_argument("--db", default="data/db/quant.duckdb")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    max_polls = None if args.max_polls == 0 else args.max_polls

    with DuckDBStore(args.db) as store:
        rows = poll_forever(
            symbols,
            store,
            poll_interval_sec=args.interval,
            use_fixture=(args.mode == "fixture"),
            max_polls=max_polls,
        )
        print(f"Ingested {rows} rows across {symbols}. Total rows in DB: {store.row_count()}")


if __name__ == "__main__":
    main()
