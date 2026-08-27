"""Event-driven backtest engine.

Chronologically replays option-chain snapshots (from DuckDB-ingested
history or backtest/synthetic_ticks.py) through the same production
components used live:

    tick -> alpha.quote_engine.QuoteEngine -> backtest.order_book fills
         -> risk.portfolio.Portfolio -> risk.hedging_engine
         -> backtest.performance mark-to-market

At each tick t (t > 0): quotes posted at tick t-1 are checked for fills
against tick t's touch (backtest/order_book.py's simplified snapshot-driven
fill model), fills update the portfolio and cash, the vol surface is
refit from tick t's chain, portfolio Greeks are recomputed, any risk-limit
breach triggers a hedging-engine futures order (executed at
backtest/execution_sim.py's aggressive-fill price), and the portfolio is
marked to market. Fresh quotes are then posted from tick t's data, to be
checked against tick t+1.

Quantities are unitless (not mapped to real NSE lot sizes) -- this is a
research/demo backtester over synthetic or fixture-derived data, not a
production execution simulator; see docs/WHITEPAPER.md for scope notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from alpha.quote_engine import ContractKey, Quote, QuoteEngine, QuoteParams
from backtest.execution_sim import SlippageModel, aggressive_fill_price
from backtest.order_book import MarketTouch, RestingQuote, Side, check_fill, queue_ahead_at_post
from core.option_chain import OptionChainSnapshot
from core.pricer_bindings import price as bs_price
from core.svi_surface import VolSurface, fit_surface_from_chain
from risk.hedging_engine import HedgeConfig, HedgeOrder, compute_hedge_orders
from risk.portfolio import GreeksSnapshot, Portfolio, Position
from risk.risk_limits import RiskLimits

logger = logging.getLogger(__name__)

# Minimum fraction of contract-ticks that must carry a real (bid > 0 and
# ask > 0) quote before the engine will run against a dataset by default.
# Synthetic ticks (backtest/synthetic_ticks.py) and genuine intraday quote
# feeds sit near 100%; EOD-only sources like NSE Bhavcopy sit at exactly
# 0% (bid=ask=0 for every row -- see docs/WHITEPAPER.md). This threshold
# just needs to cleanly separate those two cases, not be finely tuned.
MIN_QUOTE_COVERAGE = 0.5


class InsufficientQuoteDataError(RuntimeError):
    """Raised when the input snapshots don't carry real bid/ask quotes.

    backtest/order_book.py's fill model checks whether the next tick's
    touch crosses a resting quote's price; with bid=ask=0 (EOD settlement
    data, not a quote feed), "next tick's best_ask <= my bid" is trivially
    true for every positive bid price, producing a torrent of spurious
    fills and meaningless P&L rather than an error -- which is worse, not
    better, since it looks like a normal backtest result. This exception
    turns that silent-garbage failure mode into a loud, actionable one.
    """


def quote_coverage(snapshots_by_symbol: dict[str, list[OptionChainSnapshot]]) -> float:
    """Fraction of (symbol, tick, contract) rows with a real bid/ask quote."""
    total = 0
    quoted = 0
    for snapshots in snapshots_by_symbol.values():
        for snapshot in snapshots:
            for c in snapshot.contracts:
                total += 1
                if c.bid > 0 and c.ask > 0:
                    quoted += 1
    return quoted / total if total else 0.0


@dataclass(slots=True, frozen=True)
class BacktestConfig:
    quote_size: float = 10.0
    quote_params: QuoteParams = field(default_factory=QuoteParams)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    hedge_config: HedgeConfig = field(default_factory=HedgeConfig)
    slippage: SlippageModel = field(default_factory=SlippageModel)


@dataclass(slots=True, frozen=True)
class TradeRecord:
    timestamp: datetime
    symbol: str
    kind: str  # "quote_fill" or "hedge"
    side: str
    price: float
    quantity: float
    detail: str = ""


@dataclass(slots=True)
class BacktestResult:
    trade_log: list[TradeRecord] = field(default_factory=list)
    pnl_curve: list[tuple[datetime, float]] = field(default_factory=list)
    net_delta_history: list[float] = field(default_factory=list)
    final_portfolio: Portfolio | None = None
    final_cash: float = 0.0


class BacktestEngine:
    def __init__(self, config: BacktestConfig = BacktestConfig()) -> None:
        self.config = config
        self.quote_engine = QuoteEngine(config.quote_params)
        self.portfolio = Portfolio()
        self.cash = 0.0

        self._resting: dict[tuple[ContractKey, Side], RestingQuote] = {}
        self._last_touch: dict[ContractKey, MarketTouch] = {}
        self._surfaces: dict[str, VolSurface] = {}
        self._spots: dict[str, float] = {}
        self._futures_prices: dict[str, float] = {}

        self.result = BacktestResult()

    def run(
        self, snapshots_by_symbol: dict[str, list[OptionChainSnapshot]], allow_quoteless_data: bool = False
    ) -> BacktestResult:
        coverage = quote_coverage(snapshots_by_symbol)
        if coverage < MIN_QUOTE_COVERAGE and not allow_quoteless_data:
            raise InsufficientQuoteDataError(
                f"Only {coverage:.0%} of contract-ticks carry a real bid/ask quote (need >= "
                f"{MIN_QUOTE_COVERAGE:.0%}). This backtester's fill model needs genuine touch data; "
                "EOD-only sources (e.g. NSE Bhavcopy) report bid=ask=0 for every row, which produces "
                "spurious 'crossed the touch' fills and meaningless P&L (see docs/WHITEPAPER.md §7). "
                "Use synthetic tick data (backtest/run_backtest.py --source synthetic) or real intraday "
                "quotes once available (Phase 2 broker adapters) instead. Pass allow_quoteless_data=True "
                "(--allow-quoteless-data on the CLI) only if you understand the resulting numbers are not "
                "meaningful and want to see the mechanics run anyway."
            )

        ticks: list[tuple[datetime, str, OptionChainSnapshot]] = [
            (snap.timestamp, symbol, snap) for symbol, snaps in snapshots_by_symbol.items() for snap in snaps
        ]
        ticks.sort(key=lambda t: t[0])

        for ts, symbol, snapshot in ticks:
            self._process_tick(ts, symbol, snapshot)

        self.result.final_portfolio = self.portfolio
        self.result.final_cash = self.cash
        return self.result

    def _process_tick(self, ts: datetime, symbol: str, snapshot: OptionChainSnapshot) -> None:
        self._resolve_fills(snapshot)

        self._spots[symbol] = snapshot.spot
        self._futures_prices[symbol] = snapshot.spot
        self._surfaces[symbol] = fit_surface_from_chain(snapshot, valuation_time=ts)

        self._run_hedging(ts)
        self._post_quotes(snapshot, ts)

        mtm = self._mark_to_market(ts)
        self.result.pnl_curve.append((ts, mtm))

        greeks = self.portfolio.compute_greeks(self._surfaces, self._spots, ts)
        self.result.net_delta_history.append(greeks.total("delta"))

    def _resolve_fills(self, snapshot: OptionChainSnapshot) -> None:
        for c in snapshot.contracts:
            key: ContractKey = (c.symbol, c.expiry, c.strike, c.option_type)
            prev_touch = self._last_touch.get(key)
            volume_delta = (c.volume - prev_touch.cumulative_volume) if prev_touch else 0.0
            touch = MarketTouch(best_bid=c.bid, best_ask=c.ask, bid_qty=c.bid_qty, ask_qty=c.ask_qty, cumulative_volume=c.volume)

            for side in (Side.BUY, Side.SELL):
                quote = self._resting.pop((key, side), None)
                if quote is None:
                    continue
                filled_qty, fill_price = check_fill(quote, touch, max(volume_delta, 0.0))
                if filled_qty <= 0:
                    continue
                signed_qty = filled_qty if side == Side.BUY else -filled_qty
                self.portfolio.add_position(
                    Position(symbol=c.symbol, quantity=signed_qty, expiry=c.expiry, strike=c.strike, option_type=c.option_type)
                )
                self.cash -= signed_qty * fill_price
                self.result.trade_log.append(
                    TradeRecord(
                        timestamp=c.timestamp,
                        symbol=c.symbol,
                        kind="quote_fill",
                        side=side.value,
                        price=fill_price,
                        quantity=filled_qty,
                        detail=f"{c.strike}{c.option_type.value} {c.expiry}",
                    )
                )

            self._last_touch[key] = touch

    def _run_hedging(self, ts: datetime) -> None:
        greeks: GreeksSnapshot = self.portfolio.compute_greeks(self._surfaces, self._spots, ts)
        orders: list[HedgeOrder] = compute_hedge_orders(
            greeks, self.config.risk_limits, self._futures_prices, self.config.hedge_config
        )
        for order in orders:
            spot = self._futures_prices.get(order.symbol)
            if spot is None:
                continue
            fill_price = aggressive_fill_price(spot, order.quantity > 0, order.quantity, self.config.slippage)
            self.portfolio.add_position(Position(symbol=order.symbol, quantity=order.quantity))
            self.cash -= order.quantity * fill_price
            self.result.trade_log.append(
                TradeRecord(
                    timestamp=ts,
                    symbol=order.symbol,
                    kind="hedge",
                    side="BUY" if order.quantity > 0 else "SELL",
                    price=fill_price,
                    quantity=abs(order.quantity),
                    detail=order.reason,
                )
            )

    def _post_quotes(self, snapshot: OptionChainSnapshot, ts: datetime) -> None:
        surface = self._surfaces.get(snapshot.symbol)
        quotes: list[Quote] = self.quote_engine.quote_snapshot(
            snapshot, surface=surface, portfolio=self.portfolio, valuation_time=ts
        )
        contracts_by_key = {(c.symbol, c.expiry, c.strike, c.option_type): c for c in snapshot.contracts}

        for q in quotes:
            c = contracts_by_key[q.contract_key]
            self._resting[(q.contract_key, Side.BUY)] = RestingQuote(
                side=Side.BUY,
                price=q.bid,
                quantity=self.config.quote_size,
                queue_ahead=queue_ahead_at_post(Side.BUY, q.bid, c.bid, c.ask, c.bid_qty, c.ask_qty),
                posted_at=ts,
            )
            self._resting[(q.contract_key, Side.SELL)] = RestingQuote(
                side=Side.SELL,
                price=q.ask,
                quantity=self.config.quote_size,
                queue_ahead=queue_ahead_at_post(Side.SELL, q.ask, c.bid, c.ask, c.bid_qty, c.ask_qty),
                posted_at=ts,
            )

    def _mark_to_market(self, ts: datetime) -> float:
        value = self.cash
        for symbol in self.portfolio.symbols():
            surface = self._surfaces.get(symbol)
            spot = self._spots.get(symbol)
            value += self.portfolio.net_underlying_position(symbol) * (spot or 0.0)
            if surface is None or spot is None:
                continue
            for pos in self.portfolio.positions:
                if pos.symbol != symbol or pos.is_future:
                    continue
                T = max((pos.expiry - ts.date()).days, 0) / 365.0
                if T <= 0:
                    continue
                sigma = surface.iv(pos.expiry, pos.strike)
                is_call = pos.option_type.value == "CE"
                theo = bs_price(spot, pos.strike, T, 0.065, 0.065, sigma, is_call)
                value += pos.quantity * theo
        return value
