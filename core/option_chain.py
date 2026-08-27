"""Common data structures shared by every market-data adapter, the pricer,
alpha, risk and backtest layers. Every adapter (nsepython, Shoonya, Upstox)
normalizes into these types so downstream code never needs to know which
data source produced a snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"


@dataclass(slots=True)
class OptionContract:
    """A single strike/expiry/option-type quote at a point in time."""

    symbol: str  # underlying, e.g. "NIFTY" / "BANKNIFTY"
    expiry: date
    strike: float
    option_type: OptionType
    ltp: float
    bid: float
    bid_qty: int
    ask: float
    ask_qty: int
    oi: int
    change_in_oi: int
    volume: int
    timestamp: datetime
    iv: float | None = None  # exchange-reported IV, if available; the pricer recomputes its own

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.ltp

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0


@dataclass(slots=True)
class OptionChainSnapshot:
    """A full option chain pull for one underlying at one point in time."""

    symbol: str
    timestamp: datetime
    spot: float
    contracts: list[OptionContract] = field(default_factory=list)

    @property
    def expiries(self) -> list[date]:
        return sorted({c.expiry for c in self.contracts})

    def contracts_for_expiry(self, expiry: date) -> list[OptionContract]:
        return [c for c in self.contracts if c.expiry == expiry]
