"""Manual, explicit smoke test for a live Shoonya (Finvasia) NorenApi login +
websocket subscription -- the Phase 2 analogue of
scripts/live_pull_smoke_test.py (which does the same job for nsepython).

This answers one narrow question safely -- "does a live Shoonya
login+subscribe actually work with these credentials?" -- before
`main.py --mode live --broker shoonya` is trusted for a longer-running
session.

Safety properties
------------------
  - Never runs implicitly. `--yes-hit-live-shoonya` is a required flag;
    there is no default that reaches the network.
  - Credentials are read ONLY from environment variables, never CLI args
    (args land in shell history and `ps`, env vars don't automatically).
    Missing variables fail fast with a message naming which ones -- never
    their values.
  - Exactly one login attempt and one subscribe call; no retry loop (the
    adapter's own `_run_forever` reconnect-with-backoff is for a real
    long-running session, deliberately not used here).
  - Never prints the session token, password hash, or any other credential
    material -- only a short, sanitized summary or error class/message.
  - Writes results to a dedicated `data/db/shoonya_live_smoke_test.duckdb`.
    Never touches the synthetic dev DB, checked-in fixtures, or anything
    tests/ depends on.

Usage
-----
    export SHOONYA_USER_ID=... SHOONYA_PASSWORD=... SHOONYA_TOTP_SECRET=...
    export SHOONYA_VENDOR_CODE=... SHOONYA_API_KEY=...
    python scripts/shoonya_live_smoke_test.py --yes-hit-live-shoonya --symbol NIFTY
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.shoonya_ws_adapter import ShoonyaCredentials, ShoonyaWebSocketAdapter  # noqa: E402
from data.duckdb_store import DuckDBStore  # noqa: E402

SAFE_DB_PATH = "data/db/shoonya_live_smoke_test.duckdb"
MAX_ERROR_MESSAGE_LEN = 300

_REQUIRED_ENV_VARS = (
    "SHOONYA_USER_ID",
    "SHOONYA_PASSWORD",
    "SHOONYA_TOTP_SECRET",
    "SHOONYA_VENDOR_CODE",
    "SHOONYA_API_KEY",
)


def credentials_from_env() -> ShoonyaCredentials:
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "Missing required environment variable(s): " + ", ".join(missing) + ". "
            "Set them (see this script's module docstring) before running "
            "--yes-hit-live-shoonya. Refusing to proceed with partial/placeholder credentials."
        )
    return ShoonyaCredentials(
        user_id=os.environ["SHOONYA_USER_ID"],
        password=os.environ["SHOONYA_PASSWORD"],
        totp_secret=os.environ["SHOONYA_TOTP_SECRET"],
        vendor_code=os.environ["SHOONYA_VENDOR_CODE"],
        api_key=os.environ["SHOONYA_API_KEY"],
        imei=os.environ.get("SHOONYA_IMEI", "quant-smoke-test"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--yes-hit-live-shoonya",
        action="store_true",
        required=True,
        help="Required acknowledgement that this makes a real login + websocket connection to Shoonya",
    )
    parser.add_argument("--symbol", default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    parser.add_argument(
        "--timeout-sec", type=float, default=30.0, help="How long to wait for at least one tick after subscribing"
    )
    args = parser.parse_args()

    credentials = credentials_from_env()
    adapter = ShoonyaWebSocketAdapter(credentials)

    print(f"[shoonya-smoke-test] logging in as {credentials.user_id} and opening the websocket...")
    start = time.monotonic()
    try:
        adapter.connect()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_MESSAGE_LEN]
        print(f"[shoonya-smoke-test] FAILED to connect after {time.monotonic() - start:.1f}s: {message}")
        sys.exit(1)
    print(f"[shoonya-smoke-test] connected in {time.monotonic() - start:.1f}s")

    first_snapshot = threading.Event()

    def on_snapshot(snapshot) -> None:
        first_snapshot.set()

    try:
        print(f"[shoonya-smoke-test] subscribing to {args.symbol} (resolving symbol master)...")
        adapter.subscribe([args.symbol], on_snapshot)

        if not first_snapshot.wait(timeout=args.timeout_sec):
            print(f"[shoonya-smoke-test] FAILED: no tick received within {args.timeout_sec:.0f}s")
            print(
                "[shoonya-smoke-test] note: this can be normal outside NSE trading hours "
                "(09:15-15:30 IST, weekdays) -- Shoonya's feed is quiet when the market is closed."
            )
            sys.exit(1)

        snapshot = adapter.get_option_chain(args.symbol)
        print(
            f"[shoonya-smoke-test] SUCCESS: symbol={snapshot.symbol} spot={snapshot.spot} "
            f"contracts={len(snapshot.contracts)}"
        )

        with DuckDBStore(SAFE_DB_PATH) as store:
            n = store.insert_snapshot(snapshot)
            print(f"[shoonya-smoke-test] wrote {n} rows to {SAFE_DB_PATH} (row_count={store.row_count(args.symbol)})")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_MESSAGE_LEN]
        print(f"[shoonya-smoke-test] FAILED: {message}")
        sys.exit(1)
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
