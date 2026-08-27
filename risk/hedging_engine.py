"""Automated delta/gamma hedging: when a net exposure breaches its
configured risk limit, size a futures order that brings the desk back
inside a *partial*-hedge band (not fully flat) net of estimated bid-ask and
slippage cost.

Hedging fully back to zero on every breach would over-trade -- paying the
futures bid-ask spread and slippage on every small oscillation around the
limit. Instead, `target_band_fraction` hedges back to a fraction of the
limit itself (e.g. 0.5 = hedge back to 50% of max_net_delta), leaving
headroom before the next hedge trade is triggered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from risk.portfolio import GreeksSnapshot
from risk.risk_limits import RiskLimits, check_breaches


@dataclass(slots=True, frozen=True)
class HedgeConfig:
    target_band_fraction: float = 0.5  # hedge back to this fraction of the breached limit
    futures_half_spread: float = 0.5  # index points, one-way cost of crossing the futures bid-ask
    slippage_bps: float = 1.0  # additional market-impact slippage, in bps of notional


@dataclass(slots=True, frozen=True)
class HedgeOrder:
    symbol: str
    quantity: float  # signed futures quantity: positive = buy, negative = sell
    reason: str
    estimated_cost: float  # bid-ask + slippage cost estimate, in the underlying's currency


def _estimate_cost(quantity: float, price: float | None, config: HedgeConfig) -> float:
    spread_cost = abs(quantity) * config.futures_half_spread
    notional = abs(quantity) * price if price is not None else 0.0
    slippage_cost = notional * config.slippage_bps / 10_000.0
    return spread_cost + slippage_cost


def compute_hedge_orders(
    greeks: GreeksSnapshot,
    limits: RiskLimits,
    futures_prices: dict[str, float],
    config: HedgeConfig = HedgeConfig(),
) -> list[HedgeOrder]:
    """Only delta breaches generate a hedge order here: futures are a pure
    delta instrument (gamma/vega of a futures position are zero), so a
    gamma or vega breach is reported by risk_limits.check_breaches for the
    desk to act on via the options book itself (e.g. reduce the position or
    trade an offsetting option), not something a futures hedge can fix.
    """
    orders: list[HedgeOrder] = []
    for breach in check_breaches(greeks, limits):
        if breach.greek != "delta":
            continue
        target = math.copysign(breach.limit * config.target_band_fraction, breach.value)
        hedge_qty = -(breach.value - target)
        if abs(hedge_qty) < 1e-9:
            continue
        orders.append(
            HedgeOrder(
                symbol=breach.symbol,
                quantity=hedge_qty,
                reason=str(breach),
                estimated_cost=_estimate_cost(hedge_qty, futures_prices.get(breach.symbol), config),
            )
        )
    return orders
