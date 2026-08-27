"""Manual, explicit smoke test for a single live NSE option-chain pull.

This is deliberately separate from `data/ingest.py`'s `--mode live` (which
is meant for real operational polling once live access is confirmed
working). This script exists to answer one narrow question safely --
"does a live pull actually work from here?" -- before anything is wired
into a longer-running loop.

Safety properties
------------------
  - Never runs implicitly. `--yes-hit-live-nse` is a required flag; there
    is no default that reaches the network.
  - Exactly one attempt, no retry loop. `nsepython.nsefetch` (vendored,
    not part of this project) makes up to three sequential HTTPS GETs
    (nseindia.com homepage, the option-chain page, then the API call)
    each with its own hardcoded 10s timeout to establish session cookies
    NSE's anti-bot check expects -- that is the library's behavior, not
    something this script controls or retries on top of.
  - No secrets or credentials are involved: `nsepython` scrapes NSE's
    public option-chain JSON with browser-like headers, the same way a
    browser tab would; there is no API key, login, or personal data in
    the request or response.
  - Never prints raw HTTP headers, cookies, or full tracebacks (which
    could otherwise leak local proxy/network configuration into logs) --
    only a short, sanitized summary or error class/message.
  - Writes results to a dedicated `data/db/live_smoke_test.duckdb`. Never
    touches the synthetic dev DB, the checked-in fixture JSON files under
    data/sample_data/, or anything tests/ depends on.

Usage
-----
    python scripts/live_pull_smoke_test.py --yes-hit-live-nse --symbol NIFTY
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.nse_python_adapter import NSEPythonAdapter  # noqa: E402
from data.duckdb_store import DuckDBStore  # noqa: E402

SAFE_DB_PATH = "data/db/live_smoke_test.duckdb"
MAX_ERROR_MESSAGE_LEN = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--yes-hit-live-nse",
        action="store_true",
        required=True,
        help="Required acknowledgement that this makes a real outbound HTTPS request to nseindia.com",
    )
    parser.add_argument("--symbol", default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    args = parser.parse_args()

    print(f"[live-smoke-test] attempting ONE live nsepython pull for {args.symbol}...")
    print("[live-smoke-test] up to 3 sequential HTTPS requests, 10s timeout each (nsepython's own behavior)")

    adapter = NSEPythonAdapter(use_fixture=False)
    adapter.connect()

    start = time.monotonic()
    try:
        snapshot = adapter.get_option_chain(args.symbol)
    except Exception as exc:
        elapsed = time.monotonic() - start
        message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_MESSAGE_LEN]
        print(f"[live-smoke-test] FAILED after {elapsed:.1f}s: {message}")
        print(
            "[live-smoke-test] note: a KeyError on 'records' here usually means NSE returned an "
            "empty/non-JSON response -- most often their anti-bot check blocking a non-browser "
            "client or a datacenter IP, not necessarily a bug in this codebase."
        )
        sys.exit(1)
    finally:
        adapter.disconnect()

    elapsed = time.monotonic() - start
    print(f"[live-smoke-test] SUCCESS in {elapsed:.1f}s")
    print(
        f"  symbol={snapshot.symbol} spot={snapshot.spot} timestamp={snapshot.timestamp} "
        f"contracts={len(snapshot.contracts)}"
    )

    with DuckDBStore(SAFE_DB_PATH) as store:
        n = store.insert_snapshot(snapshot)
        print(f"[live-smoke-test] wrote {n} rows to {SAFE_DB_PATH} (row_count={store.row_count(args.symbol)})")


if __name__ == "__main__":
    main()
