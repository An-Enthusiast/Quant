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

`--diagnostic` mode replicates `nsepython.nsefetch`'s own three-step
session sequence (homepage -> option-chain page -> API call) directly
with `requests`, but -- unlike `nsefetch`, which silently swallows a
JSON-decode failure into `{}` -- reports each step's HTTP status code and
a short, truncated snippet of the final response body. This is still a
single pass (one request per step, no retries); it exists only to tell
apart "blocked before reaching NSE," "NSE returned a non-200," and "NSE
returned a challenge/CAPTCHA page instead of JSON."

Usage
-----
    python scripts/live_pull_smoke_test.py --yes-hit-live-nse --symbol NIFTY
    python scripts/live_pull_smoke_test.py --yes-hit-live-nse --diagnostic
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
MAX_BODY_SNIPPET_LEN = 250

# Same session-priming sequence as nsepython.rahu.nsefetch (local mode):
# two GETs to establish cookies NSE's front door expects, then the API call.
_DIAGNOSTIC_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
}


def run_diagnostic(symbol: str) -> None:
    import requests

    session = requests.Session()
    steps = [
        ("homepage", "https://www.nseindia.com"),
        ("option-chain page", "https://www.nseindia.com/option-chain"),
        ("api call", f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"),
    ]
    for label, url in steps:
        try:
            resp = session.get(url, headers=_DIAGNOSTIC_HEADERS, timeout=10)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_MESSAGE_LEN]
            print(f"[diagnostic] {label}: FAILED before a response was received -- {message}")
            return

        content_type = resp.headers.get("content-type", "?")
        print(f"[diagnostic] {label}: status={resp.status_code} content-type={content_type} bytes={len(resp.content)}")

        if label == "api call":
            snippet = resp.text[:MAX_BODY_SNIPPET_LEN].replace("\n", " ").replace("\r", "")
            print(f"[diagnostic] response body snippet: {snippet!r}")
            if resp.status_code == 200 and content_type.startswith("application/json"):
                print("[diagnostic] looks like valid JSON was returned -- the normal pull should work")
            else:
                print(
                    "[diagnostic] not a 200 JSON response -- this is NSE's own front door rejecting the "
                    "request (bot/WAF check, rate limiting, or similar), not a local network block"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--yes-hit-live-nse",
        action="store_true",
        required=True,
        help="Required acknowledgement that this makes a real outbound HTTPS request to nseindia.com",
    )
    parser.add_argument("--symbol", default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Report each step's HTTP status/body instead of doing the normal (opaque) pull",
    )
    args = parser.parse_args()

    if args.diagnostic:
        print(f"[diagnostic] replicating nsepython's session sequence for {args.symbol} (one pass, no retries)...")
        run_diagnostic(args.symbol)
        return

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
