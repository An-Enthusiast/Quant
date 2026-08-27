"""Phase 2 hook: Upstox websocket adapter (v3 market-data feed, protobuf
encoded) for live L2 tick streaming.

NOT YET IMPLEMENTED -- mirror of `core/shoonya_ws_adapter.py`'s scaffold for
Upstox. See that module's docstring for the shared rationale; this file
documents the Upstox-specific pieces.

To complete this adapter:
  1. Implement `connect()`: OAuth2 login (client id/secret + redirect) to
     obtain an access token, then GET the authorized websocket URI from
     Upstox's `/feed/market-data-feed/authorize` endpoint and open it.
  2. Compile Upstox's `MarketDataFeedV3.proto` (published in their API docs)
     with `protoc` into `_upstox_feed_pb2.py`, and subscribe to `full` mode
     for the option instrument keys of the requested underlying's active
     expiries.
  3. In `_on_message`, deserialize the binary protobuf frame
     (`FeedResponse.ParseFromString(raw_bytes)`), walk `response.feeds[key]`
     for each instrument, and map its `ltpc`/`marketFF`/`optionGreeks` /
     depth fields onto `OptionContract` -- Upstox's `full` feed already
     reports greeks and IV server-side, which can be used as a
     cross-check against this project's own `core/pricer_bindings.py`
     output.
  4. Implement reconnect-with-backoff in `_run_forever` (skeleton below).
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
class UpstoxCredentials:
    client_id: str
    client_secret: str
    redirect_uri: str
    access_token: str | None = None  # populated by the OAuth2 flow in connect()


class UpstoxProtobufAdapter(MarketDataInterface):
    """Live L2 tick adapter over Upstox's protobuf market-data feed. Same
    `MarketDataInterface` contract as `NSEPythonAdapter`/`ShoonyaWebSocketAdapter`.
    """

    def __init__(self, credentials: UpstoxCredentials, max_reconnect_backoff_sec: float = 30.0) -> None:
        self.credentials = credentials
        self.max_reconnect_backoff_sec = max_reconnect_backoff_sec
        self._latest_snapshots: dict[str, OptionChainSnapshot] = {}
        self._callbacks: list[Callable[[OptionChainSnapshot], None]] = []

    @property
    def is_live(self) -> bool:
        return True

    def connect(self) -> None:
        raise NotImplementedError(
            "UpstoxProtobufAdapter.connect() requires Upstox API credentials and an OAuth2 "
            "access token. Implement the authorize + websocket-URI handshake here (see module "
            "docstring)."
        )

    def disconnect(self) -> None:
        self.credentials.access_token = None

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
            "UpstoxProtobufAdapter.subscribe() requires connect() to have established a live "
            "websocket session first. Implement instrument-key subscription + protobuf "
            "_on_message tick mapping here (see module docstring)."
        )

    def _on_message(self, raw_bytes: bytes) -> None:
        """Where the binary protobuf `FeedResponse` frame gets decoded and
        mapped onto OptionContract / OptionChainSnapshot, then dispatched to
        `self._callbacks`. Left unimplemented pending Upstox credentials and
        the compiled `_upstox_feed_pb2` module.
        """
        raise NotImplementedError

    def _run_forever(self) -> None:
        backoff = 1.0
        while True:
            try:
                self.connect()
                backoff = 1.0
            except Exception:
                logger.exception("UpstoxProtobufAdapter: connection failed, retrying in %.1fs", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff_sec)
