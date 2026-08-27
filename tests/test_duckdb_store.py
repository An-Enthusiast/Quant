"""Tests for the DuckDB ingestion store (data/duckdb_store.py)."""

from __future__ import annotations

from data.duckdb_store import DuckDBStore
from core.nse_python_adapter import NSEPythonAdapter


def _sample_snapshot():
    adapter = NSEPythonAdapter(use_fixture=True)
    adapter.connect()
    return adapter.get_option_chain("NIFTY")


def test_insert_and_query_roundtrip(tmp_path):
    snapshot = _sample_snapshot()
    db_path = tmp_path / "test.duckdb"

    with DuckDBStore(db_path) as store:
        n_inserted = store.insert_snapshot(snapshot)
        assert n_inserted == len(snapshot.contracts)

        df = store.query_range("NIFTY")
        assert len(df) == len(snapshot.contracts)
        assert set(df["symbol"].unique()) == {"NIFTY"}
        assert store.row_count("NIFTY") == len(snapshot.contracts)
        assert store.row_count("BANKNIFTY") == 0


def test_distinct_timestamps(tmp_path):
    snapshot = _sample_snapshot()
    db_path = tmp_path / "test.duckdb"

    with DuckDBStore(db_path) as store:
        store.insert_snapshot(snapshot)
        timestamps = store.distinct_timestamps("NIFTY")
        assert timestamps == [snapshot.timestamp]


def test_query_range_filters_by_expiry(tmp_path):
    snapshot = _sample_snapshot()
    db_path = tmp_path / "test.duckdb"
    expiry = snapshot.expiries[0]

    with DuckDBStore(db_path) as store:
        store.insert_snapshot(snapshot)
        df = store.query_range("NIFTY", expiry=expiry)
        assert len(df) == len(snapshot.contracts_for_expiry(expiry))
        assert (df["expiry"].dt.date == expiry).all()


def test_empty_snapshot_inserts_zero_rows(tmp_path):
    from datetime import datetime

    from core.option_chain import OptionChainSnapshot

    db_path = tmp_path / "test.duckdb"
    empty = OptionChainSnapshot(symbol="NIFTY", timestamp=datetime.now(), spot=25000.0, contracts=[])
    with DuckDBStore(db_path) as store:
        assert store.insert_snapshot(empty) == 0
        assert store.row_count() == 0
