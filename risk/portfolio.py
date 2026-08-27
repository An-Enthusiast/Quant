"""Multi-asset (Nifty + BankNifty) non-linear portfolio risk aggregation.

Tracks every open option position plus any underlying-futures hedge
positions, and aggregates net Delta/Gamma/Theta/Vega per underlying (and in
total) using the Numba-vectorized Greeks engine (core/greeks_numba.py) --
one batched array call across every option position of a given underlying,
rather than a per-contract Python loop, which is what actually matters once
a desk is carrying positions across the full Nifty + BankNifty chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from core.greeks_numba import vectorized_greeks
from core.option_chain import OptionType
from core.svi_surface import VolSurface


@dataclass(slots=True, frozen=True)
class Position:
    symbol: str
    quantity: float  # signed: positive = long, negative = short
    expiry: date | None = None  # None => this is an underlying/futures position
    strike: float | None = None
    option_type: OptionType | None = None

    @property
    def is_future(self) -> bool:
        return self.option_type is None


@dataclass(slots=True, frozen=True)
class GreeksSnapshot:
    net_delta: dict[str, float] = field(default_factory=dict)
    net_gamma: dict[str, float] = field(default_factory=dict)
    net_theta: dict[str, float] = field(default_factory=dict)
    net_vega: dict[str, float] = field(default_factory=dict)

    def total(self, greek: str) -> float:
        return sum(getattr(self, f"net_{greek}").values())


@dataclass(slots=True)
class Portfolio:
    positions: list[Position] = field(default_factory=list)

    def add_position(self, position: Position) -> None:
        self.positions.append(position)

    def get_position(self, symbol: str, expiry: date, strike: float, option_type: OptionType) -> float:
        """Net signed quantity in one specific contract -- used by
        alpha/quote_engine.py to compute the Avellaneda-Stoikov inventory
        skew for that contract.
        """
        return sum(
            p.quantity
            for p in self.positions
            if p.symbol == symbol and p.expiry == expiry and p.strike == strike and p.option_type == option_type
        )

    def net_underlying_position(self, symbol: str) -> float:
        return sum(p.quantity for p in self.positions if p.symbol == symbol and p.is_future)

    def symbols(self) -> list[str]:
        return sorted({p.symbol for p in self.positions})

    def compute_greeks(
        self,
        surfaces: dict[str, VolSurface],
        spots: dict[str, float],
        valuation_time: datetime,
        r: float = 0.065,
        q: float = 0.065,
    ) -> GreeksSnapshot:
        """Aggregates net Delta/Gamma/Theta/Vega per underlying.

        `surfaces`/`spots` are keyed by underlying symbol -- typically the
        latest fitted `VolSurface` (core/svi_surface.py) and spot price for
        each of NIFTY / BANKNIFTY. A position whose underlying has no entry
        in `surfaces` is skipped (can't be priced) and logged implicitly via
        the returned snapshot simply omitting that symbol's Greeks.
        """
        snapshot = GreeksSnapshot()

        for symbol in self.symbols():
            future_qty = self.net_underlying_position(symbol)
            option_positions = [p for p in self.positions if p.symbol == symbol and not p.is_future]

            net_delta = float(future_qty)  # 1 futures contract = 1 unit of delta
            net_gamma = net_theta = net_vega = 0.0

            if option_positions:
                surface = surfaces.get(symbol)
                spot = spots.get(symbol)
                if surface is not None and spot is not None:
                    n = len(option_positions)
                    S = np.full(n, spot)
                    K = np.array([p.strike for p in option_positions], dtype=np.float64)
                    T = np.array(
                        [max((p.expiry - valuation_time.date()).days, 0) / 365.0 for p in option_positions],
                        dtype=np.float64,
                    )
                    sigma = np.array(
                        [surface.iv(p.expiry, p.strike) for p in option_positions], dtype=np.float64
                    )
                    is_call = np.array(
                        [p.option_type == OptionType.CALL for p in option_positions], dtype=np.bool_
                    )
                    R = np.full(n, r)
                    Q = np.full(n, q)
                    qty = np.array([p.quantity for p in option_positions], dtype=np.float64)

                    valid = T > 0
                    if valid.any():
                        delta, gamma, theta, vega, _rho = vectorized_greeks(
                            S[valid], K[valid], T[valid], R[valid], Q[valid], sigma[valid], is_call[valid]
                        )
                        w = qty[valid]
                        net_delta += float(np.dot(delta, w))
                        net_gamma += float(np.dot(gamma, w))
                        net_theta += float(np.dot(theta, w))
                        net_vega += float(np.dot(vega, w))

            snapshot.net_delta[symbol] = net_delta
            snapshot.net_gamma[symbol] = net_gamma
            snapshot.net_theta[symbol] = net_theta
            snapshot.net_vega[symbol] = net_vega

        return snapshot
