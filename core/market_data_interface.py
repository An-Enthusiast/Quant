"""`MarketDataInterface` -- the abstraction that lets the pricer, alpha and
risk layers stay completely ignorant of where ticks come from.

Rollout plan:
  Phase 1 (zero-cost prototyping): `NSEPythonAdapter` polls
      `nsepython.nse_optionchain_scrapper` for a JSON option-chain snapshot.
      No API keys, broker account, or paid subscription required.
  Phase 2 (live streaming): `ShoonyaWebSocketAdapter` / `UpstoxProtobufAdapter`
      push L2 tick updates over a broker websocket.

Every adapter implements the same four methods below and yields the same
`OptionChainSnapshot` / `OptionContract` types (core/option_chain.py), so
switching from Phase 1 to Phase 2 is a one-line change in main.py -- no
change to core/pricer_bindings.py, alpha/quote_engine.py, risk/*, or
backtest/* is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from core.option_chain import OptionChainSnapshot


class MarketDataInterface(ABC):
    """Abstract base for all option-chain data sources."""

    @abstractmethod
    def connect(self) -> None:
        """Establish the underlying connection (HTTP session, websocket, etc.)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection cleanly."""

    @abstractmethod
    def get_option_chain(self, symbol: str) -> OptionChainSnapshot:
        """Synchronous pull-based fetch of the latest full chain snapshot.

        Always available (Phase 1 polling adapters implement this natively;
        Phase 2 streaming adapters implement it by returning their most
        recently cached snapshot for `symbol`).
        """

    @abstractmethod
    def subscribe(self, symbols: list[str], callback: Callable[[OptionChainSnapshot], None]) -> None:
        """Register `callback` to be invoked on every new snapshot for `symbols`.

        Phase 1 polling adapters implement this by calling `callback` at the
        end of each poll cycle. Phase 2 streaming adapters implement it by
        wiring `callback` into their websocket message handler.
        """

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for tick-level streaming adapters, False for polling adapters.

        Downstream consumers (e.g. the quoting engine's toxicity features)
        use this to decide how much to trust intra-snapshot microstructure
        signals derived from polling-cadence deltas.
        """
