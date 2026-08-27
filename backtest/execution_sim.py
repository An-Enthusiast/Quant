"""Execution cost model for aggressive (market) orders -- used by the
hedging engine's futures orders, which cross the spread immediately rather
than resting passively (passive-quote fills are handled by
backtest/order_book.py's queue model instead).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SlippageModel:
    half_spread: float = 0.5  # index points, one-way cost of crossing the futures bid-ask
    impact_bps_per_unit: float = 0.02  # additional linear market-impact slippage, bps of price per unit traded


def aggressive_fill_price(mid: float, is_buy: bool, quantity: float, model: SlippageModel = SlippageModel()) -> float:
    """Execution price for an aggressive order of `quantity` at `mid`:
    crosses half the spread immediately, plus a linear market-impact term
    proportional to size (larger hedge trades move the market more).
    """
    impact = mid * (model.impact_bps_per_unit / 10_000.0) * abs(quantity)
    if is_buy:
        return mid + model.half_spread + impact
    return mid - model.half_spread - impact
