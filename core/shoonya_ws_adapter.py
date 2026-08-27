"""Phase 2 hook: Shoonya (Finvasia) websocket adapter for live L2 tick
streaming.

NOT YET IMPLEMENTED -- this is the scaffold a future PR fills in once a
Shoonya broker account + API credentials are available. It exists now so
that:
  1. `main.py --mode live --broker shoonya` has a concrete target to select.
  2. The exact seam between broker-specific tick JSON and the shared
     `OptionChainSnapshot`/`OptionContract` types (core/option_chain.py) is
     documented in one place, so the pricer/alpha/risk layers never need to
     change when this adapter goes live.

To complete this adapter:
  1. Implement `connect()`: authenticate via the Shoonya REST login flow
     (user id, password, TOTP, vendor/app key -> session token), then open
     the websocket named in `ShoonyaCredentials.ws_endpoint` using that
     token.
  2. Subscribe to the NFO option-chain touchline/depth feed for each
     instrument token belonging to the requested underlying's active
     expiries (Shoonya's `NorenApi.subscribe` / `touchline` topics).
  3. In `_on_message`, map Shoonya's tick schema (`tk`, `e`, `lp`, `bp1`,
     `sp1`, `bq1`, `sq1`, `oi`, ...) onto `OptionContract` fields and push
     an updated `OptionChainSnapshot` (merging the tick into the last known
     full chain for that underlying+expiry) to every registered callback.
  4. Implement reconnect-with-backoff in `_run_forever` (the outer loop
     below already has the retry skeleton).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from core.market_data_interface import MarketDataInterface
from core.option_chain import OptionChainSnapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ShoonyaCredentials:
    user_id: str
    password: str
    totp_secret: str
    vendor_code: str
    api_key: str
    imei: str
    ws_endpoint: str = "wss://api.shoonya.com/NorenWSTP/"


class ShoonyaWebSocketAdapter(MarketDataInterface):
    """Live L2 tick adapter over Shoonya's websocket feed. Same
    `MarketDataInterface` contract as `NSEPythonAdapter` -- swapping this in
    for Phase 2 requires zero changes to `core/pricer_bindings.py`,
    `alpha/quote_engine.py`, `risk/*`, or `backtest/*`.
    """

    def __init__(self, credentials: ShoonyaCredentials, max_reconnect_backoff_sec: float = 30.0) -> None:
        self.credentials = credentials
        self.max_reconnect_backoff_sec = max_reconnect_backoff_sec
        self._session_token: str | None = None
        self._latest_snapshots: dict[str, OptionChainSnapshot] = {}
        self._callbacks: list[Callable[[OptionChainSnapshot], None]] = []

    @property
    def is_live(self) -> bool:
        return True

    def connect(self) -> None:
        raise NotImplementedError(
            "ShoonyaWebSocketAdapter.connect() requires a Shoonya broker account and API "
            "credentials. Implement the NorenApi login handshake here (see module docstring)."
        )

    def disconnect(self) -> None:
        self._session_token = None

    def get_option_chain(self, symbol: str) -> OptionChainSnapshot:
        try:
            return self._latest_snapshots[symbol]
        except KeyError as exc:
            raise RuntimeError(
                f"No snapshot cached yet for {symbol}; subscribe() must receive at least one "
                "tick before get_option_chain() can return data."
            ) from exc

    def subscribe(self, symbols: list[str], callback: Callable[[OptionChainSnapshot], None]) -> None:
        self._callbacks.append(callback)
        raise NotImplementedError(
            "ShoonyaWebSocketAdapter.subscribe() requires connect() to have established a live "
            "websocket session first. Implement instrument-token subscription + _on_message "
            "tick mapping here (see module docstring)."
        )

    def _on_message(self, raw_tick: dict) -> None:
        """Where broker-specific tick JSON gets mapped onto OptionContract /
        OptionChainSnapshot and dispatched to `self._callbacks`. Left
        unimplemented pending Shoonya credentials.
        """
        raise NotImplementedError

    def _run_forever(self) -> None:
        backoff = 1.0
        while True:
            try:
                self.connect()
                backoff = 1.0
            except Exception:
                logger.exception("ShoonyaWebSocketAdapter: connection failed, retrying in %.1fs", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff_sec)
