"""CLI orchestrator for populating DuckDB with option-chain snapshots.

Four modes:
  fixture        -- one-shot ingest from the recorded sample chain (no network).
  live           -- continuous nsepython polling. NSE's own edge currently
                     rejects nsepython's scraped requests (403/404, see
                     docs/WHITEPAPER.md) -- kept for whenever that changes, or
                     for use against a less bot-gated deployment.
  bhavcopy       -- backfills real historical daily EOD data from NSE's public
                     Bhavcopy archive over the network (data/bhavcopy_loader.py).
                     Confirmed reliably reachable (no bot-blocking) -- currently
                     the best source of *real* NSE data this project has.
  bhavcopy-local -- ingests real Bhavcopy days from a local directory of
                     pre-fetched CSVs (data/sample_data/bhavcopy_history/,
                     built with `python -m data.bhavcopy_loader build-archive`)
                     -- zero network calls, for environments where nseindia.com
                     isn't reachable at all (e.g. some cloud sandboxes; see
                     docs/WHITEPAPER.md).

Examples
--------
    python -m data.ingest --mode fixture --symbols NIFTY,BANKNIFTY
    python -m data.ingest --mode live --symbols NIFTY,BANKNIFTY --interval 5 --max-polls 0
    python -m data.ingest --mode bhavcopy --symbols NIFTY,BANKNIFTY --lookback-days 10
    python -m data.ingest --mode bhavcopy --start-date 2026-08-01 --end-date 2026-08-26
    python -m data.ingest --mode bhavcopy-local --symbols NIFTY,BANKNIFTY
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

from data.bhavcopy_loader import find_latest_available_bhavcopy, ingest_bhavcopy_range, ingest_local_archive
from data.duckdb_store import DuckDBStore
from data.nsepython_poller import poll_forever

DEFAULT_LOCAL_ARCHIVE_DIR = "data/sample_data/bhavcopy_history"


def _run_bhavcopy(args: argparse.Namespace, symbols: list[str], store: DuckDBStore) -> None:
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end, _ = find_latest_available_bhavcopy(max_lookback_days=args.lookback_days)
        start = end - timedelta(days=args.lookback_days)

    rows = ingest_bhavcopy_range(store, start, end, symbols, request_delay_sec=args.request_delay)
    print(f"Bhavcopy-ingested {rows} rows across {symbols} for {start}..{end}. Total rows in DB: {store.row_count()}")


def _run_bhavcopy_local(args: argparse.Namespace, symbols: list[str], store: DuckDBStore) -> None:
    rows = ingest_local_archive(store, args.local_dir, symbols)
    print(
        f"Bhavcopy-local-ingested {rows} rows across {symbols} from {args.local_dir} "
        f"(no network used). Total rows in DB: {store.row_count()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NSE option-chain snapshots into DuckDB")
    parser.add_argument("--mode", choices=["fixture", "live", "bhavcopy", "bhavcopy-local"], default="fixture")
    parser.add_argument("--symbols", default="NIFTY,BANKNIFTY", help="Comma-separated underlyings")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between polls (live mode)")
    parser.add_argument(
        "--max-polls", type=int, default=1, help="Number of poll rounds; 0 means run forever (live mode)"
    )
    parser.add_argument("--lookback-days", type=int, default=10, help="bhavcopy mode: days to backfill")
    parser.add_argument("--start-date", default=None, help="bhavcopy mode: YYYY-MM-DD (overrides --lookback-days)")
    parser.add_argument("--end-date", default=None, help="bhavcopy mode: YYYY-MM-DD")
    parser.add_argument("--request-delay", type=float, default=0.5, help="bhavcopy mode: seconds between requests")
    parser.add_argument(
        "--local-dir", default=DEFAULT_LOCAL_ARCHIVE_DIR, help="bhavcopy-local mode: directory of bhavcopy_*.csv files"
    )
    parser.add_argument("--db", default="data/db/quant.duckdb")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    with DuckDBStore(args.db) as store:
        if args.mode == "bhavcopy":
            _run_bhavcopy(args, symbols, store)
            return
        if args.mode == "bhavcopy-local":
            _run_bhavcopy_local(args, symbols, store)
            return

        max_polls = None if args.max_polls == 0 else args.max_polls
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
