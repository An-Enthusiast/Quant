r"""Avellaneda & Stoikov (2008), "High-frequency trading in a limit order
book" -- optimal market-making reservation price and spread.

Derivation sketch
------------------
A risk-averse market maker with CARA utility U(x) = -exp(-gamma*x) holds
inventory q in an asset whose mid price follows arithmetic Brownian motion
dS_t = sigma dW_t, and can post limit orders whose fill probability decays
exponentially in distance from mid: lambda(delta) = A * exp(-kappa*delta).
Maximizing expected utility of terminal wealth via the HJB equation and
applying the CARA/Gaussian indifference-price argument yields, at time t
with T the terminal horizon:

Reservation (indifference) price -- the price at which the maker is
indifferent between holding and not holding one more unit of inventory:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)                    (1)

  A market maker long inventory (q > 0) skews r *below* the mid s (eager
  to sell, reluctant to buy more); short inventory (q < 0) skews r
  *above* s. Quotes are then centered on r, not s -- this is the whole
  point of inventory-aware market making versus naive symmetric quoting.

Optimal total spread (sum of the maker's distance from mid on each side):

    delta_a + delta_b = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)   (2)

  The first term is the same inventory-risk premium as in (1) (spread
  must widen as risk aversion, volatility, or time-to-close grow); the
  second term is a liquidity premium trading off fill probability against
  markup, governed by kappa (higher kappa = order flow arrives from
  closer to mid = tighter competitive spread).

Individual bid/ask quotes are r -+ (spread/2); see `compute_quote` below.
Inventory-skew beyond (1) and toxicity-driven spread widening on top of
(2) are applied downstream in alpha/quote_engine.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def reservation_price(mid: float, inventory: float, gamma: float, sigma: float, time_to_close: float) -> float:
    """Eq. (1): r(s, q, t) = s - q * gamma * sigma^2 * (T - t)."""
    return mid - inventory * gamma * sigma**2 * time_to_close


def optimal_spread(gamma: float, sigma: float, time_to_close: float, kappa: float) -> float:
    """Eq. (2): total (bid+ask) distance from the reservation price."""
    if gamma <= 0:
        raise ValueError("gamma (risk aversion) must be positive")
    if kappa <= 0:
        raise ValueError("kappa (order-arrival decay) must be positive")
    inventory_term = gamma * sigma**2 * max(time_to_close, 0.0)
    liquidity_term = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
    return inventory_term + liquidity_term


@dataclass(slots=True, frozen=True)
class ASQuote:
    reservation_price: float
    half_spread: float

    @property
    def bid(self) -> float:
        return self.reservation_price - self.half_spread

    @property
    def ask(self) -> float:
        return self.reservation_price + self.half_spread


def compute_quote(
    mid: float, inventory: float, gamma: float, sigma: float, time_to_close: float, kappa: float
) -> ASQuote:
    """Combines (1) and (2) into a bid/ask quote centered on the
    reservation price. `inventory` is the maker's current signed position
    in this contract (positive = long); `time_to_close` is in years (e.g.
    time remaining to the trading session close or to expiry, whichever
    horizon the desk is managing risk against).
    """
    r = reservation_price(mid, inventory, gamma, sigma, time_to_close)
    spread = optimal_spread(gamma, sigma, time_to_close, kappa)
    return ASQuote(reservation_price=r, half_spread=spread / 2.0)
