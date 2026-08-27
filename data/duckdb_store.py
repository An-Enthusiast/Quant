"""DuckDB-backed store for option-chain snapshots.

Batches inserts through a pandas DataFrame + `duckdb`'s native DataFrame
scan (`INSERT INTO ... SELECT * FROM df`) rather than row-by-row `INSERT`,
and never loads the full history into memory for a write -- each
`insert_snapshot` call only ever holds one snapshot's rows (a few hundred
at most) in memory, while queries push filtering down into DuckDB itself
so callers can pull an arbitrary-sized time range without the store
buffering it all up front.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from core.option_chain import OptionChainSnapshot

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DuckDBStore:
    def __init__(self, db_path: str | Path = "data/db/quant.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))
        self._conn.execute(_SCHEMA_PATH.read_text())

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def insert_snapshot(self, snapshot: OptionChainSnapshot) -> int:
        """Appends every contract in `snapshot` as one row each. Returns the
        number of rows written.
        """
        if not snapshot.contracts:
            return 0
        ingested_at = datetime.now()
        rows = [
            {
                "ts": snapshot.timestamp,
                "ingested_at": ingested_at,
                "symbol": c.symbol,
                "expiry": c.expiry,
                "strike": c.strike,
                "option_type": c.option_type.value,
                "underlying_spot": snapshot.spot,
                "ltp": c.ltp,
                "bid": c.bid,
                "bid_qty": c.bid_qty,
                "ask": c.ask,
                "ask_qty": c.ask_qty,
                "oi": c.oi,
                "change_in_oi": c.change_in_oi,
                "volume": c.volume,
                "exchange_iv": c.iv,
            }
            for c in snapshot.contracts
        ]
        df = pd.DataFrame(rows)
        self._conn.execute("INSERT INTO option_chain_snapshots SELECT * FROM df")
        logger.info("DuckDBStore: inserted %d rows for %s @ %s", len(rows), snapshot.symbol, snapshot.timestamp)
        return len(rows)

    def query_range(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        expiry: date | None = None,
    ) -> pd.DataFrame:
        """Pulls rows for `symbol` in [start, end], filtering push-down into
        DuckDB. Returns a pandas DataFrame -- the caller decides how much of
        the range to materialize.
        """
        clauses = ["symbol = ?"]
        params: list[object] = [symbol]
        if start is not None:
            clauses.append("ts >= ?")
            params.append(start)
        if end is not None:
            clauses.append("ts <= ?")
            params.append(end)
        if expiry is not None:
            clauses.append("expiry = ?")
            params.append(expiry)
        where = " AND ".join(clauses)
        query = f"SELECT * FROM option_chain_snapshots WHERE {where} ORDER BY ts, expiry, strike, option_type"  # noqa: S608
        return self._conn.execute(query, params).fetchdf()

    def distinct_timestamps(self, symbol: str) -> list[datetime]:
        rows = self._conn.execute(
            "SELECT DISTINCT ts FROM option_chain_snapshots WHERE symbol = ? ORDER BY ts", [symbol]
        ).fetchall()
        return [r[0] for r in rows]

    def row_count(self, symbol: str | None = None) -> int:
        if symbol is None:
            return self._conn.execute("SELECT COUNT(*) FROM option_chain_snapshots").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM option_chain_snapshots WHERE symbol = ?", [symbol]
        ).fetchone()[0]
