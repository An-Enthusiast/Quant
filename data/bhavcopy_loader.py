"""NSE F&O Bhavcopy (UDiFF end-of-day archive) loader.

Unlike the live scraped option-chain API (core/nse_python_adapter.py),
which NSE's own edge/WAF rejects for non-browser clients (see
docs/WHITEPAPER.md), the daily Bhavcopy archive at `nsearchives.nseindia.com`
is a plain public static file server -- confirmed reachable with a single
unauthenticated GET, no cookies, no bot-check headers, no retries needed.
It publishes one CSV (zipped) per trading day covering every NSE F&O
instrument: close/settlement price, open interest, change in OI, traded
volume and value, for every strike/expiry/option-type.

URL scheme (NSE's UDiFF format, replacing the pre-2024-07-08 Bhav Copy):

    https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip

Trade-off versus the live/intraday adapters: Bhavcopy has no bid/ask --
it is an end-of-day settlement report, not a quote feed. This module
reuses `OptionContract`/`OptionChainSnapshot` (core/option_chain.py)
rather than inventing a parallel type, relying on their existing
"no bid/ask known" fallback (`bid=ask=0` -> `.mid` falls back to `.ltp`,
`.spread` reports `0.0` honestly rather than a fabricated number) --
`ltp` is set to the day's close price. That makes Bhavcopy-derived
snapshots correctly usable for SVI surface fitting and Greeks/risk
analytics on *real* historical data, while remaining honestly unusable
for the tick-level quoting/fill backtester, which genuinely needs bid/ask
microstructure that this data source does not have.
"""

from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path

import requests

from core.option_chain import OptionChainSnapshot, OptionContract, OptionType

logger = logging.getLogger(__name__)

_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "accept": "*/*",
}
_OPTION_INSTRUMENT_TYPES = {"IDO", "STO"}  # index option, stock option
_EOD_TIME = dt_time(15, 30)  # NSE F&O market close


def bhavcopy_url(trade_date: date) -> str:
    return _URL_TEMPLATE.format(date_str=trade_date.strftime("%Y%m%d"))


def download_bhavcopy_zip(trade_date: date, timeout: float = 20.0) -> bytes:
    """Single unauthenticated GET for one day's Bhavcopy zip. Raises
    `FileNotFoundError` for a 404 (weekend/holiday/no-file-yet -- the
    normal, expected case for most calendar dates), or `requests`'s own
    exception for a genuine network failure.
    """
    url = bhavcopy_url(trade_date)
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    if resp.status_code == 404:
        raise FileNotFoundError(f"No Bhavcopy published for {trade_date} ({url})")
    resp.raise_for_status()
    return resp.content


def find_latest_available_bhavcopy(
    reference_date: date | None = None, max_lookback_days: int = 10
) -> tuple[date, bytes]:
    """Walks backward from `reference_date` (default: today) looking for
    the most recent trading day with a published Bhavcopy. This is NOT a
    retry-on-failure loop -- weekends/holidays are the normal, expected
    reason a given date has no file, and there is no separate NSE
    trading-calendar API to consult first, so scanning backward a bounded
    number of days is the standard, necessary way to find "the last
    trading day." Raises `FileNotFoundError` if nothing is found within
    `max_lookback_days`.
    """
    reference_date = reference_date or date.today()
    for i in range(max_lookback_days):
        trade_date = reference_date - timedelta(days=i)
        try:
            content = download_bhavcopy_zip(trade_date)
        except FileNotFoundError:
            continue
        return trade_date, content
    raise FileNotFoundError(
        f"No Bhavcopy found in the {max_lookback_days} days up to {reference_date}"
    )


def _extract_csv_text(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return io.TextIOWrapper(f, encoding="utf-8").read()


def parse_bhavcopy_csv_text(
    csv_text: str, symbols: list[str], trade_date: date | None = None
) -> dict[str, OptionChainSnapshot]:
    """Pure parsing function (no I/O) -- unit-tested directly against a
    recorded sample (data/sample_data/fo_bhavcopy_sample.csv), the same
    separation-of-concerns pattern as
    core.nse_python_adapter.parse_option_chain_payload.

    Builds one `OptionChainSnapshot` per requested symbol from that
    symbol's index-option (`IDO`) rows. `trade_date` overrides the date
    parsed from the CSV's own `TradDt` column, for callers that already
    know it (e.g. a fixture with a fixed recorded date).
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    wanted = set(symbols)
    by_symbol: dict[str, list[OptionContract]] = {s: [] for s in wanted}
    spot_by_symbol: dict[str, float] = {}
    parsed_trade_date: date | None = None

    for row in reader:
        symbol = row["TckrSymb"]
        if symbol not in wanted or row["FinInstrmTp"] not in _OPTION_INSTRUMENT_TYPES:
            continue

        if parsed_trade_date is None:
            parsed_trade_date = datetime.strptime(row["TradDt"], "%Y-%m-%d").date()

        strike = float(row["StrkPric"])
        option_type = OptionType(row["OptnTp"])
        close = float(row["ClsPric"])
        settle = float(row["SttlmPric"])
        ltp = close if close > 0 else settle

        oi = int(float(row["OpnIntrst"]))
        change_oi = int(float(row["ChngInOpnIntrst"]))
        volume = int(float(row["TtlTradgVol"]))
        spot = float(row["UndrlygPric"])
        spot_by_symbol.setdefault(symbol, spot)

        expiry = datetime.strptime(row["XpryDt"], "%Y-%m-%d").date()
        timestamp = datetime.combine(trade_date or parsed_trade_date, _EOD_TIME)

        by_symbol[symbol].append(
            OptionContract(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                ltp=ltp,
                bid=0.0,
                bid_qty=0,
                ask=0.0,
                ask_qty=0,
                oi=oi,
                change_in_oi=change_oi,
                volume=volume,
                timestamp=timestamp,
                iv=None,  # Bhavcopy doesn't report IV; recomputed from ltp by core.svi_surface
            )
        )

    timestamp = datetime.combine(trade_date or parsed_trade_date or date.today(), _EOD_TIME)
    return {
        symbol: OptionChainSnapshot(symbol=symbol, timestamp=timestamp, spot=spot_by_symbol[symbol], contracts=contracts)
        for symbol, contracts in by_symbol.items()
        if contracts
    }


def load_bhavcopy_snapshots(trade_date: date, symbols: list[str]) -> dict[str, OptionChainSnapshot]:
    """Downloads and parses one day's Bhavcopy for `symbols`. Raises
    `FileNotFoundError` if that date has no published file.
    """
    zip_bytes = download_bhavcopy_zip(trade_date)
    csv_text = _extract_csv_text(zip_bytes)
    return parse_bhavcopy_csv_text(csv_text, symbols, trade_date=trade_date)


def ingest_bhavcopy_range(
    store,
    start_date: date,
    end_date: date,
    symbols: list[str],
    request_delay_sec: float = 0.5,
) -> int:
    """Ingests every available trading day in `[start_date, end_date]`
    (inclusive) into `store` (data.duckdb_store.DuckDBStore). Days with no
    published file (weekends/holidays) are skipped, not treated as
    errors. `request_delay_sec` is a small politeness delay between
    requests to a public archive -- not required by any documented rate
    limit, just good citizenship for a bulk multi-day pull.
    """
    total_rows = 0
    trade_date = start_date
    while trade_date <= end_date:
        try:
            snapshots = load_bhavcopy_snapshots(trade_date, symbols)
        except FileNotFoundError:
            logger.info("bhavcopy_loader: no file for %s (weekend/holiday), skipping", trade_date)
            trade_date += timedelta(days=1)
            continue

        for snapshot in snapshots.values():
            total_rows += store.insert_snapshot(snapshot)

        trade_date += timedelta(days=1)
        if trade_date <= end_date:
            time.sleep(request_delay_sec)

    return total_rows


# --- Local archive: pre-fetch real Bhavcopy days to plain CSV files on disk,
# so a network-restricted environment (e.g. a cloud sandbox whose egress
# policy blocks nseindia.com -- see docs/WHITEPAPER.md) can still ingest
# real historical data without hitting the network at all. -----------------

_LOCAL_ARCHIVE_FILENAME_TEMPLATE = "bhavcopy_{date_str}.csv"
_LOCAL_ARCHIVE_FILENAME_RE_PREFIX = "bhavcopy_"


def _local_archive_filename(trade_date: date) -> str:
    return _LOCAL_ARCHIVE_FILENAME_TEMPLATE.format(date_str=trade_date.strftime("%Y%m%d"))


def save_filtered_bhavcopy_csv(trade_date: date, symbols: list[str], out_dir: str | Path) -> Path | None:
    """Downloads one day's Bhavcopy, filters rows to `symbols`' index
    option/future instrument rows, and writes the result as a plain
    (unzipped) CSV to `out_dir` -- small enough to check into git (a
    day's NIFTY+BANKNIFTY subset is a few hundred KB, versus the ~1MB
    zipped full-market file). Returns the written path, or `None` if
    `trade_date` has no published file (weekend/holiday -- not an error).
    """
    out_dir = Path(out_dir)
    try:
        zip_bytes = download_bhavcopy_zip(trade_date)
    except FileNotFoundError:
        return None

    csv_text = _extract_csv_text(zip_bytes)
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames
    wanted = set(symbols)
    kept_types = _OPTION_INSTRUMENT_TYPES | {"IDF", "STF"}  # options + their underlying futures
    rows = [row for row in reader if row["TckrSymb"] in wanted and row["FinInstrmTp"] in kept_types]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _local_archive_filename(trade_date)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def build_local_archive(
    out_dir: str | Path, symbols: list[str], start_date: date, end_date: date, request_delay_sec: float = 0.5
) -> list[date]:
    """Calls `save_filtered_bhavcopy_csv` for every day in
    `[start_date, end_date]`. Returns the trading days actually saved
    (weekends/holidays are silently skipped, not errors).
    """
    saved: list[date] = []
    trade_date = start_date
    while trade_date <= end_date:
        path = save_filtered_bhavcopy_csv(trade_date, symbols, out_dir)
        if path is not None:
            saved.append(trade_date)
            logger.info("bhavcopy_loader: saved %s", path)
        else:
            logger.info("bhavcopy_loader: no file for %s (weekend/holiday), skipping", trade_date)
        trade_date += timedelta(days=1)
        if trade_date <= end_date:
            time.sleep(request_delay_sec)
    return saved


def load_local_archive(directory: str | Path, symbols: list[str]) -> dict[str, list[OptionChainSnapshot]]:
    """Reads every `bhavcopy_*.csv` file in `directory` (as saved by
    `save_filtered_bhavcopy_csv`/`build_local_archive`) and parses them
    into a chronological list of `OptionChainSnapshot` per symbol. Pure
    local file I/O -- no network involved, so this works in any
    environment regardless of NSE access.
    """
    directory = Path(directory)
    by_symbol: dict[str, list[OptionChainSnapshot]] = {s: [] for s in symbols}

    for path in sorted(directory.glob(f"{_LOCAL_ARCHIVE_FILENAME_RE_PREFIX}*.csv")):
        snapshots = parse_bhavcopy_csv_text(path.read_text(), symbols)
        for symbol, snapshot in snapshots.items():
            by_symbol[symbol].append(snapshot)

    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda snap: snap.timestamp)

    return by_symbol


def ingest_local_archive(store, directory: str | Path, symbols: list[str]) -> int:
    """Loads a local archive (see `load_local_archive`) and inserts every
    snapshot into `store`. Returns the total row count inserted.
    """
    total_rows = 0
    for snapshots in load_local_archive(directory, symbols).values():
        for snapshot in snapshots:
            total_rows += store.insert_snapshot(snapshot)
    return total_rows


def _cli() -> None:
    """`python -m data.bhavcopy_loader build-archive ...` -- fetches a real
    date range from NSE (network required) and saves it as plain CSVs
    under a local directory, for later network-free ingestion via
    `python -m data.ingest --mode bhavcopy-local` (see that module's
    docstring).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build a local Bhavcopy archive of plain CSV files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-archive", help="Download and save a date range as local CSVs")
    build.add_argument("--symbols", default="NIFTY,BANKNIFTY")
    build.add_argument("--out-dir", default="data/sample_data/bhavcopy_history")
    build.add_argument("--lookback-days", type=int, default=35, help="Ignored if --start-date/--end-date given")
    build.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    build.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    build.add_argument("--request-delay", type=float, default=0.5)
    build.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end, _ = find_latest_available_bhavcopy(max_lookback_days=args.lookback_days)
        start = end - timedelta(days=args.lookback_days)

    saved = build_local_archive(args.out_dir, symbols, start, end, request_delay_sec=args.request_delay)
    print(f"Saved {len(saved)} trading days ({start}..{end}) to {args.out_dir}: {[d.isoformat() for d in saved]}")


if __name__ == "__main__":
    _cli()
