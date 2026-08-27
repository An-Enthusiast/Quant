"""Tests for the backtest engine's quote-coverage guard
(backtest/event_engine.py::quote_coverage / InsufficientQuoteDataError).

Regression coverage for the exact failure mode found running the
backtester against real 1-month Bhavcopy (EOD, no bid/ask) data: the fill
model treated a bid=ask=0 touch as "crossed" for every positive-priced
quote, producing tens of thousands of spurious fills and a P&L in the
tens of millions -- silently, with no error, on a completed run. The
guard turns that into a loud, actionable failure by default.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from backtest.event_engine import BacktestConfig, BacktestEngine, InsufficientQuoteDataError, quote_coverage
from backtest.synthetic_ticks import generate_tick_series
from core.option_chain import OptionChainSnapshot, OptionContract, OptionType
from risk.risk_limits import RiskLimits


def _eod_snapshot(symbol: str, day: datetime, spot: float) -> OptionChainSnapshot:
    expiry = day.date() + timedelta(days=7)
    contracts = [
        OptionContract(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            option_type=opt_type,
            ltp=100.0 + i,
            bid=0.0,
            bid_qty=0,
            ask=0.0,
            ask_qty=0,
            oi=1000,
            change_in_oi=0,
            volume=100,
            timestamp=day,
        )
        for i, strike in enumerate([spot - 100, spot, spot + 100])
        for opt_type in (OptionType.CALL, OptionType.PUT)
    ]
    return OptionChainSnapshot(symbol=symbol, timestamp=day, spot=spot, contracts=contracts)


def _eod_series(symbol: str, spot: float, n_days: int = 5) -> list[OptionChainSnapshot]:
    start = datetime(2026, 7, 22, 15, 30)
    return [_eod_snapshot(symbol, start + timedelta(days=i), spot) for i in range(n_days)]


def _synthetic_series(n_ticks: int = 20) -> list[OptionChainSnapshot]:
    expiry = date.today().fromordinal(date.today().toordinal() + 7)
    return generate_tick_series(
        "NIFTY", spot0=25000.0, strike_step=50.0, base_iv=0.13, expiry=expiry, n_ticks=n_ticks, tick_interval_sec=5.0
    )


def test_quote_coverage_zero_for_eod_only_data():
    snapshots = {"NIFTY": _eod_series("NIFTY", 25000.0)}
    assert quote_coverage(snapshots) == 0.0


def test_quote_coverage_high_for_synthetic_tick_data():
    snapshots = {"NIFTY": _synthetic_series()}
    # Deep-OTM synthetic contracts can occasionally round to a zero bid, so
    # this isn't asserted at exactly 1.0 -- just well above the guard's
    # MIN_QUOTE_COVERAGE threshold.
    assert quote_coverage(snapshots) > 0.9


def test_backtest_engine_refuses_eod_only_data_by_default():
    snapshots = {"NIFTY": _eod_series("NIFTY", 25000.0)}
    engine = BacktestEngine(BacktestConfig(risk_limits=RiskLimits(max_net_delta={"NIFTY": 300.0})))
    try:
        engine.run(snapshots)
        raise AssertionError("expected InsufficientQuoteDataError")
    except InsufficientQuoteDataError as exc:
        assert "bid/ask" in str(exc)


def test_backtest_engine_runs_eod_data_when_explicitly_allowed():
    snapshots = {"NIFTY": _eod_series("NIFTY", 25000.0)}
    engine = BacktestEngine(BacktestConfig(risk_limits=RiskLimits(max_net_delta={"NIFTY": 300.0})))
    result = engine.run(snapshots, allow_quoteless_data=True)
    assert len(result.pnl_curve) == len(snapshots["NIFTY"])


def test_backtest_engine_unaffected_for_normal_synthetic_data():
    snapshots = {"NIFTY": _synthetic_series()}
    engine = BacktestEngine(BacktestConfig(risk_limits=RiskLimits(max_net_delta={"NIFTY": 300.0})))
    result = engine.run(snapshots)  # must NOT raise
    assert len(result.pnl_curve) == len(snapshots["NIFTY"])
