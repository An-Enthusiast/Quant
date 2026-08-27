"""Per-underlying risk-limit configuration and breach detection against a
`risk.portfolio.GreeksSnapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from risk.portfolio import GreeksSnapshot

_DEFAULT_LIMITS = {"NIFTY": None, "BANKNIFTY": None}


@dataclass(slots=True, frozen=True)
class RiskLimits:
    """`None` for any symbol means "no limit configured" -- that Greek is
    monitored but never triggers a breach/hedge for that underlying.
    """

    max_net_delta: dict[str, float | None] = field(default_factory=lambda: dict(_DEFAULT_LIMITS))
    max_net_gamma: dict[str, float | None] = field(default_factory=lambda: dict(_DEFAULT_LIMITS))
    max_net_vega: dict[str, float | None] = field(default_factory=lambda: dict(_DEFAULT_LIMITS))


@dataclass(slots=True, frozen=True)
class Breach:
    symbol: str
    greek: str
    value: float
    limit: float

    def __str__(self) -> str:
        return f"{self.symbol} net {self.greek} breach: {self.value:.2f} exceeds limit {self.limit:.2f}"


def check_breaches(greeks: GreeksSnapshot, limits: RiskLimits) -> list[Breach]:
    breaches: list[Breach] = []
    for greek_name, values, limit_map in (
        ("delta", greeks.net_delta, limits.max_net_delta),
        ("gamma", greeks.net_gamma, limits.max_net_gamma),
        ("vega", greeks.net_vega, limits.max_net_vega),
    ):
        for symbol, value in values.items():
            limit = limit_map.get(symbol)
            if limit is not None and abs(value) > limit:
                breaches.append(Breach(symbol=symbol, greek=greek_name, value=value, limit=limit))
    return breaches
