"""QuoteEngine: produces final bid/ask quotes for every contract in an
option-chain snapshot by combining:

  1. The Avellaneda-Stoikov reservation price + optimal spread
     (alpha/avellaneda_stoikov.py), using the desk's current inventory in
     each contract (risk/portfolio.py) and that contract's SVI-surface
     implied vol (core/svi_surface.py).
  2. Toxicity-score-driven spread widening: `spread *= (1 + beta * score)`,
     capped at `max_toxicity_widen`. `toxicity_score` defaults to 0 (no
     widening) for any contract with no score supplied -- the quoting
     logic is fully functional standalone, without a trained toxicity
     model wired in, per this project's rollout (see
     alpha/toxicity_model.py).

Toxicity scores are computed elsewhere (they need a sequential feature
history a single snapshot doesn't have -- see alpha/features.py,
alpha/toxicity_model.py) and passed in as a `{contract_key: score}` dict by
the caller (typically backtest/event_engine.py or a live orchestration
loop), keeping this module a pure per-snapshot quoting function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from alpha.avellaneda_stoikov import compute_quote
from core.option_chain import OptionChainSnapshot, OptionType

ContractKey = tuple[str, date, float, OptionType]  # (symbol, expiry, strike, option_type)


@dataclass(slots=True, frozen=True)
class QuoteParams:
    gamma: float = 0.1  # risk aversion
    kappa: float = 1.5  # order-arrival intensity decay
    toxicity_beta: float = 2.0  # spread-widening sensitivity to toxicity score
    max_toxicity_widen: float = 5.0  # cap on the (1 + beta * score) multiplier
    min_tick: float = 0.05  # NSE index-option tick size
    fallback_sigma: float = 0.15  # used if no vol-surface slice is available for a contract


@dataclass(slots=True, frozen=True)
class Quote:
    contract_key: ContractKey
    bid: float
    ask: float
    reservation_price: float
    spread: float
    toxicity_score: float


def _round_to_tick(x: float, tick: float) -> float:
    return round(x / tick) * tick


def quote_contract(
    contract_key: ContractKey,
    mid: float,
    inventory: float,
    sigma: float,
    time_to_close: float,
    params: QuoteParams = QuoteParams(),
    toxicity_score: float = 0.0,
) -> Quote:
    base = compute_quote(mid, inventory, params.gamma, sigma, time_to_close, params.kappa)
    widen_mult = min(1.0 + params.toxicity_beta * max(toxicity_score, 0.0), params.max_toxicity_widen)
    half_spread = base.half_spread * widen_mult

    bid = _round_to_tick(base.reservation_price - half_spread, params.min_tick)
    ask = _round_to_tick(base.reservation_price + half_spread, params.min_tick)
    if ask <= bid:
        ask = bid + params.min_tick

    return Quote(
        contract_key=contract_key,
        bid=max(bid, 0.0),
        ask=ask,
        reservation_price=base.reservation_price,
        spread=ask - bid,
        toxicity_score=toxicity_score,
    )


class PortfolioLike:
    """Structural type: anything with `get_position` works (see
    risk/portfolio.py's `Portfolio`); kept minimal so tests can pass a stub
    without importing the full risk package.
    """

    def get_position(self, symbol: str, expiry: date, strike: float, option_type: OptionType) -> float: ...


@dataclass(slots=True)
class QuoteEngine:
    params: QuoteParams = field(default_factory=QuoteParams)

    def quote_snapshot(
        self,
        snapshot: OptionChainSnapshot,
        surface: object | None = None,
        portfolio: PortfolioLike | None = None,
        toxicity_scores: dict[ContractKey, float] | None = None,
        valuation_time: datetime | None = None,
    ) -> list[Quote]:
        """Quotes every contract in `snapshot`. `surface` is a
        `core.svi_surface.VolSurface` (falls back to `params.fallback_sigma`
        if omitted or a contract's expiry wasn't fit). `portfolio` is a
        `risk.portfolio.Portfolio` (falls back to zero inventory, i.e.
        naive symmetric-around-mid quoting, if omitted).
        """
        valuation_time = valuation_time or snapshot.timestamp
        toxicity_scores = toxicity_scores or {}
        quotes: list[Quote] = []

        for c in snapshot.contracts:
            T = max((c.expiry - valuation_time.date()).days, 0) / 365.0
            if T <= 0 or c.mid <= 0:
                continue

            sigma = None
            if surface is not None:
                try:
                    sigma = surface.iv(c.expiry, c.strike)
                except Exception:
                    sigma = None
            if not sigma or sigma <= 0:
                sigma = self.params.fallback_sigma

            key: ContractKey = (c.symbol, c.expiry, c.strike, c.option_type)
            inventory = portfolio.get_position(c.symbol, c.expiry, c.strike, c.option_type) if portfolio else 0.0
            toxicity_score = toxicity_scores.get(key, 0.0)

            quotes.append(quote_contract(key, c.mid, inventory, sigma, T, self.params, toxicity_score))

        return quotes
