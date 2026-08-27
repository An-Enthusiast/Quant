"""Tests for the NSE F&O Bhavcopy loader (data/bhavcopy_loader.py).

Only the pure `parse_bhavcopy_csv_text` function is exercised here,
against a recorded real sample
(data/sample_data/fo_bhavcopy_sample.csv -- trimmed from a real Bhavcopy
download to NIFTY/BANKNIFTY's front few expiries). The network-calling
functions (download_bhavcopy_zip, find_latest_available_bhavcopy,
ingest_bhavcopy_range) are intentionally not covered by the automated
suite, matching this project's convention of keeping tests network-free
(see core/nse_python_adapter.py's fixture-mode split for the same
pattern).
"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

from core.option_chain import OptionType
from core.svi_surface import fit_surface_from_chain
from data.bhavcopy_loader import bhavcopy_url, ingest_local_archive, load_local_archive, parse_bhavcopy_csv_text
from data.duckdb_store import DuckDBStore

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "fo_bhavcopy_sample.csv"
FIXTURE_TRADE_DATE = date(2026, 8, 26)
FIXTURE_TIMESTAMP = datetime.combine(FIXTURE_TRADE_DATE, time(15, 30))

LOCAL_ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "bhavcopy_history"


def _load_fixture_text() -> str:
    return FIXTURE_PATH.read_text()


def test_bhavcopy_url_format():
    url = bhavcopy_url(date(2026, 8, 26))
    assert url == "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20260826_F_0000.csv.zip"


def test_parses_both_symbols_from_fixture():
    snapshots = parse_bhavcopy_csv_text(_load_fixture_text(), ["NIFTY", "BANKNIFTY"])
    assert set(snapshots.keys()) == {"NIFTY", "BANKNIFTY"}
    for snap in snapshots.values():
        assert snap.timestamp == FIXTURE_TIMESTAMP
        assert snap.spot > 0
        assert len(snap.contracts) > 0


def test_contracts_have_no_fabricated_bid_ask():
    snapshots = parse_bhavcopy_csv_text(_load_fixture_text(), ["NIFTY"])
    for c in snapshots["NIFTY"].contracts:
        assert c.bid == 0.0
        assert c.ask == 0.0
        assert c.bid_qty == 0
        assert c.ask_qty == 0
        # mid must honestly fall back to ltp, and spread must honestly report 0 --
        # not a fabricated bid/ask-derived number (see core/option_chain.py).
        assert c.mid == c.ltp
        assert c.spread == 0.0


def test_only_requested_symbols_and_option_rows_included():
    snapshots = parse_bhavcopy_csv_text(_load_fixture_text(), ["NIFTY"])
    assert "BANKNIFTY" not in snapshots
    for c in snapshots["NIFTY"].contracts:
        assert c.symbol == "NIFTY"
        assert c.option_type in (OptionType.CALL, OptionType.PUT)


def test_unrequested_symbol_returns_empty_dict():
    snapshots = parse_bhavcopy_csv_text(_load_fixture_text(), ["RELIANCE"])
    assert snapshots == {}


def test_oi_and_volume_are_real_nonzero_values_somewhere():
    # Sanity check this is real market data, not all-zero placeholder rows.
    snapshots = parse_bhavcopy_csv_text(_load_fixture_text(), ["NIFTY", "BANKNIFTY"])
    all_contracts = [c for snap in snapshots.values() for c in snap.contracts]
    assert any(c.oi > 0 for c in all_contracts)
    assert any(c.volume > 0 for c in all_contracts)


def test_svi_surface_fits_from_bhavcopy_derived_chain():
    # End-to-end: real EOD close-price data should flow through the exact
    # same SVI-fitting pipeline as live/synthetic intraday data.
    snapshots = parse_bhavcopy_csv_text(_load_fixture_text(), ["NIFTY"])
    surface = fit_surface_from_chain(snapshots["NIFTY"], min_mid_price=0.05)
    assert len(surface.slices) >= 1
    e0 = sorted(surface.slices.keys())[0]
    atm_iv = surface.iv(e0, snapshots["NIFTY"].spot)
    assert 0.02 < atm_iv < 2.0


# --- Local archive (data/sample_data/bhavcopy_history/) -- a real,
# multi-week set of Bhavcopy days pre-fetched and checked into the repo so
# ingestion works with zero network calls (see data/bhavcopy_loader.py's
# module docstring and `python -m data.ingest --mode bhavcopy-local`). ---


def test_local_archive_directory_exists_and_has_many_real_days():
    assert LOCAL_ARCHIVE_DIR.is_dir()
    files = sorted(LOCAL_ARCHIVE_DIR.glob("bhavcopy_*.csv"))
    assert len(files) >= 20, "expected roughly a month of trading days checked in"


def test_load_local_archive_returns_chronological_real_snapshots():
    by_symbol = load_local_archive(LOCAL_ARCHIVE_DIR, ["NIFTY", "BANKNIFTY"])
    assert set(by_symbol.keys()) == {"NIFTY", "BANKNIFTY"}

    nifty_days = by_symbol["NIFTY"]
    assert len(nifty_days) >= 20
    timestamps = [snap.timestamp for snap in nifty_days]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)  # one snapshot per day, no duplicates

    for snap in nifty_days:
        assert snap.spot > 0
        assert len(snap.contracts) > 0


def test_ingest_local_archive_inserts_all_rows(tmp_path):
    db_path = tmp_path / "local_archive_test.duckdb"
    with DuckDBStore(db_path) as store:
        rows = ingest_local_archive(store, LOCAL_ARCHIVE_DIR, ["NIFTY", "BANKNIFTY"])
        assert rows > 0
        assert store.row_count() == rows
        # Real multi-day history should have more than one distinct timestamp.
        assert len(store.distinct_timestamps("NIFTY")) >= 20
