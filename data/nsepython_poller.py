"""Polling loop wiring `NSEPythonAdapter` (core/nse_python_adapter.py) into
`DuckDBStore` (data/duckdb_store.py). Works identically in fixture mode
(the current default -- no live NSE calls) and live mode (once network
access to nseindia.com is available in a given deployment); only the
`use_fixture` flag changes.
"""

from __future__ import annotations

import logging
import time

from core.nse_python_adapter import NSEPythonAdapter
from data.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)


def poll_forever(
    symbols: list[str],
    store: DuckDBStore,
    poll_interval_sec: float = 5.0,
    use_fixture: bool = False,
    max_polls: int | None = None,
) -> int:
    """Polls each symbol in `symbols` every `poll_interval_sec`, writing each
    resulting snapshot to `store`. Runs indefinitely if `max_polls` is None
    (intended for a long-lived ingestion process); stops after `max_polls`
    rounds otherwise (used by fixture-mode one-shot ingestion and tests).

    Returns the total number of rows inserted across all polls.
    """
    adapter = NSEPythonAdapter(poll_interval_sec=poll_interval_sec, use_fixture=use_fixture)
    adapter.connect()
    total_rows = 0
    polls_done = 0
    try:
        while max_polls is None or polls_done < max_polls:
            for symbol in symbols:
                try:
                    snapshot = adapter.get_option_chain(symbol)
                    total_rows += store.insert_snapshot(snapshot)
                except Exception:
                    logger.exception("nsepython_poller: failed to fetch/insert %s", symbol)
            polls_done += 1
            if max_polls is not None and polls_done >= max_polls:
                break
            time.sleep(poll_interval_sec)
    finally:
        adapter.disconnect()
    return total_rows
