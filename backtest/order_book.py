"""Resting-quote fill model for the event-driven backtester.

NSE's public option-chain snapshot (what `NSEPythonAdapter` -- and, this
session, the synthetic tick generator -- produce) reports touchline
best-bid/best-ask + resting quantity, not a full L3 order-book tape.
Without per-price-level trade prints, a textbook FIFO queue simulator isn't
reconstructable exactly, so this module uses a standard simplified
snapshot-driven fill model instead:

  - A resting quote is **fully filled** the instant the market's touch
    crosses through its price (someone traded aggressively through our
    level) -- this part is exact, not an approximation.
  - A resting quote sitting exactly *at* the touch (not crossed) is
    **partially filled** in proportion to observed traded volume at that
    price level net of the quantity we estimated was queued ahead of us
    when we posted it (`queue_ahead`, captured from the exchange-reported
    resting quantity at post time). Since true per-price trade prints
    aren't available, contract-level traded-volume delta between
    consecutive snapshots is used as a proxy for volume at that specific
    price level -- a standard approximation when only touch + cumulative
    volume are observable (see docs/WHITEPAPER.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True, frozen=True)
class RestingQuote:
    side: Side
    price: float
    quantity: float
    queue_ahead: float
    posted_at: datetime


@dataclass(slots=True, frozen=True)
class Fill:
    side: Side
    price: float
    quantity: float
    timestamp: datetime


def queue_ahead_at_post(side: Side, price: float, best_bid: float, best_ask: float, bid_qty: float, ask_qty: float) -> float:
    """Estimates resting quantity ahead of a new quote at `price`. A quote
    that improves the current touch (price-inside-the-spread) has nothing
    ahead of it; a quote joining the touch queues behind the full
    exchange-reported size at that price.
    """
    if side == Side.BUY:
        return 0.0 if price > best_bid else bid_qty
    return 0.0 if price < best_ask else ask_qty


@dataclass(slots=True, frozen=True)
class MarketTouch:
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    cumulative_volume: float


def check_fill(quote: RestingQuote, next_tick: MarketTouch, volume_delta_at_level: float) -> tuple[float, float]:
    """Checks `quote` (posted against the previous tick's touch) against
    the next tick's touch. Returns (filled_quantity, fill_price);
    filled_quantity is 0.0 if unfilled. Never fills more than
    `quote.quantity`.
    """
    if quote.side == Side.BUY:
        if next_tick.best_ask <= quote.price:
            return quote.quantity, quote.price  # market crossed through our bid: certain full fill
        if next_tick.best_bid == quote.price:
            available = max(volume_delta_at_level - quote.queue_ahead, 0.0)
            return min(quote.quantity, available), quote.price
        return 0.0, quote.price

    if next_tick.best_bid >= quote.price:
        return quote.quantity, quote.price  # market crossed through our ask: certain full fill
    if next_tick.best_ask == quote.price:
        available = max(volume_delta_at_level - quote.queue_ahead, 0.0)
        return min(quote.quantity, available), quote.price
    return 0.0, quote.price
