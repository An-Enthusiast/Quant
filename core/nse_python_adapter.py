"""Phase 1 adapter: zero-cost prototyping against NSE's public option-chain
JSON via `nsepython.nse_optionchain_scrapper`. No API keys, broker account,
or paid subscription required.

Two modes:
  - live (default): calls `nsepython.nse_optionchain_scrapper(symbol)`
    directly against nseindia.com on every poll.
  - fixture: reads a recorded JSON payload from disk instead of hitting the
    network. Used for offline development, deterministic tests, and the
    backtester, and as a drop-in stand-in until live NSE access (or a
    Phase-2 broker feed) is wired up in this deployment.

Raw payload schema (matches `nse_optionchain_scrapper`'s real output):

    {
      "records": {
        "expiryDates": ["28-Aug-2026", ...],
        "underlyingValue": 25123.45,
        "timestamp": "27-Aug-2026 15:30:01",
        "data": [
          {
            "strikePrice": 25000,
            "expiryDate": "28-Aug-2026",
            "CE": {"openInterest": ..., "changeinOpenInterest": ...,
                   "totalTradedVolume": ..., "impliedVolatility": ...,
                   "lastPrice": ..., "bidQty": ..., "bidprice": ...,
                   "askPrice": ..., "askQty": ...},
            "PE": {...}
          },
          ...
        ]
      }
    }
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from core.market_data_interface import MarketDataInterface
from core.option_chain import OptionChainSnapshot, OptionContract, OptionType

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M")
_EXPIRY_FORMAT = "%d-%b-%Y"


def _parse_timestamp(raw: str | None) -> datetime:
    if raw:
        for fmt in _TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return datetime.now()


def _parse_expiry(raw: str) -> date:
    return datetime.strptime(raw, _EXPIRY_FORMAT).date()


def parse_option_chain_payload(symbol: str, payload: dict) -> OptionChainSnapshot:
    """Pure function: raw `nse_optionchain_scrapper` JSON -> OptionChainSnapshot.

    Kept separate from the network/fixture I/O so it can be unit tested
    against a static fixture without any adapter state.
    """
    records = payload["records"]
    spot = float(records["underlyingValue"])
    timestamp = _parse_timestamp(records.get("timestamp"))

    contracts: list[OptionContract] = []
    for row in records["data"]:
        expiry = _parse_expiry(row["expiryDate"])
        strike = float(row["strikePrice"])
        for leg_key, option_type in ((("CE"), OptionType.CALL), (("PE"), OptionType.PUT)):
            leg = row.get(leg_key)
            if not leg:
                continue
            contracts.append(
                OptionContract(
                    symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    ltp=float(leg.get("lastPrice", 0.0) or 0.0),
                    bid=float(leg.get("bidprice", 0.0) or 0.0),
                    bid_qty=int(leg.get("bidQty", 0) or 0),
                    ask=float(leg.get("askPrice", 0.0) or 0.0),
                    ask_qty=int(leg.get("askQty", 0) or 0),
                    oi=int(leg.get("openInterest", 0) or 0),
                    change_in_oi=int(leg.get("changeinOpenInterest", 0) or 0),
                    volume=int(leg.get("totalTradedVolume", 0) or 0),
                    timestamp=timestamp,
                    iv=float(leg["impliedVolatility"]) if leg.get("impliedVolatility") else None,
                )
            )

    return OptionChainSnapshot(symbol=symbol, timestamp=timestamp, spot=spot, contracts=contracts)


class NSEPythonAdapter(MarketDataInterface):
    """Polling adapter over `nsepython.nse_optionchain_scrapper`.

    Parameters
    ----------
    poll_interval_sec:
        Seconds to sleep between polls in `run_forever`/`subscribe`.
    use_fixture:
        If True, `get_option_chain` reads from `fixture_dir/<symbol>.json`
        instead of calling nseindia.com. `fixture_dir` defaults to
        `data/sample_data`.
    """

    def __init__(
        self,
        poll_interval_sec: float = 5.0,
        use_fixture: bool = False,
        fixture_dir: str | Path = "data/sample_data",
    ) -> None:
        self.poll_interval_sec = poll_interval_sec
        self.use_fixture = use_fixture
        self.fixture_dir = Path(fixture_dir)
        self._connected = False

    @property
    def is_live(self) -> bool:
        return False  # polling adapter, not a tick-level stream

    def connect(self) -> None:
        self._connected = True
        logger.info(
            "NSEPythonAdapter connected (mode=%s)", "fixture" if self.use_fixture else "live-nsepython"
        )

    def disconnect(self) -> None:
        self._connected = False

    def _fetch_raw(self, symbol: str) -> dict:
        if self.use_fixture:
            fixture_path = self.fixture_dir / f"{symbol.lower()}_chain_sample.json"
            with fixture_path.open() as f:
                return json.load(f)
        # Local import: keeps `nsepython` (and its network calls) out of the
        # import graph entirely when running in fixture-only test/CI modes.
        from nsepython import nse_optionchain_scrapper

        return nse_optionchain_scrapper(symbol)

    def get_option_chain(self, symbol: str) -> OptionChainSnapshot:
        if not self._connected:
            raise RuntimeError("NSEPythonAdapter.connect() must be called before use")
        payload = self._fetch_raw(symbol)
        return parse_option_chain_payload(symbol, payload)

    def subscribe(self, symbols: list[str], callback: Callable[[OptionChainSnapshot], None]) -> None:
        """Polling loop: fetches each symbol every `poll_interval_sec` and
        invokes `callback` with the resulting snapshot. Blocks forever --
        intended to be run in its own thread/process by the orchestrator.
        """
        if not self._connected:
            raise RuntimeError("NSEPythonAdapter.connect() must be called before use")
        while True:
            for symbol in symbols:
                try:
                    callback(self.get_option_chain(symbol))
                except Exception:
                    logger.exception("NSEPythonAdapter: failed to fetch/parse %s", symbol)
            time.sleep(self.poll_interval_sec)
