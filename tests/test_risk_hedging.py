"""Tests for portfolio Greeks aggregation and the hedging engine."""

from __future__ import annotations

import pytest

from core.nse_python_adapter import NSEPythonAdapter
from core.option_chain import OptionType
from core.svi_surface import fit_surface_from_chain
from risk.hedging_engine import HedgeConfig, compute_hedge_orders
from risk.portfolio import Portfolio, Position
from risk.risk_limits import RiskLimits, check_breaches


@pytest.fixture
def nifty_snapshot_and_surface():
    adapter = NSEPythonAdapter(use_fixture=True)
    adapter.connect()
    snapshot = adapter.get_option_chain("NIFTY")
    surface = fit_surface_from_chain(snapshot)
    return snapshot, surface


def test_flat_portfolio_has_zero_greeks(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    pf = Portfolio()
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)
    assert greeks.net_delta == {}


def test_futures_position_has_unit_delta_and_zero_other_greeks(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=25.0))  # pure futures/underlying position
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)
    assert greeks.net_delta["NIFTY"] == pytest.approx(25.0)
    assert greeks.net_gamma["NIFTY"] == pytest.approx(0.0)
    assert greeks.net_vega["NIFTY"] == pytest.approx(0.0)


def test_short_call_position_gives_negative_delta(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    e0 = snapshot.expiries[0]
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=-500, expiry=e0, strike=snapshot.spot, option_type=OptionType.CALL))
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)
    assert greeks.net_delta["NIFTY"] < 0
    assert greeks.net_gamma["NIFTY"] < 0  # short options => short gamma


def test_get_position_nets_multiple_fills(nifty_snapshot_and_surface):
    snapshot, _ = nifty_snapshot_and_surface
    e0 = snapshot.expiries[0]
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=10, expiry=e0, strike=25000.0, option_type=OptionType.CALL))
    pf.add_position(Position(symbol="NIFTY", quantity=-3, expiry=e0, strike=25000.0, option_type=OptionType.CALL))
    assert pf.get_position("NIFTY", e0, 25000.0, OptionType.CALL) == 7


def test_check_breaches_detects_delta_breach(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    e0 = snapshot.expiries[0]
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=-500, expiry=e0, strike=snapshot.spot, option_type=OptionType.CALL))
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)

    limits = RiskLimits(max_net_delta={"NIFTY": 50.0})
    breaches = check_breaches(greeks, limits)
    assert any(b.greek == "delta" and b.symbol == "NIFTY" for b in breaches)


def test_no_breach_when_within_limits(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=5))
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)
    limits = RiskLimits(max_net_delta={"NIFTY": 1000.0})
    assert check_breaches(greeks, limits) == []


def test_hedge_order_sizes_toward_target_band(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    e0 = snapshot.expiries[0]
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=-500, expiry=e0, strike=snapshot.spot, option_type=OptionType.CALL))
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)

    limits = RiskLimits(max_net_delta={"NIFTY": 100.0})
    config = HedgeConfig(target_band_fraction=0.5)
    orders = compute_hedge_orders(greeks, limits, {"NIFTY": snapshot.spot}, config)

    assert len(orders) == 1
    order = orders[0]
    post_hedge_delta = greeks.net_delta["NIFTY"] + order.quantity
    assert post_hedge_delta == pytest.approx(-100.0 * config.target_band_fraction, abs=1e-6)
    assert order.estimated_cost > 0


def test_no_hedge_order_when_no_breach(nifty_snapshot_and_surface):
    snapshot, surface = nifty_snapshot_and_surface
    pf = Portfolio()
    pf.add_position(Position(symbol="NIFTY", quantity=5))
    greeks = pf.compute_greeks({"NIFTY": surface}, {"NIFTY": snapshot.spot}, snapshot.timestamp)
    limits = RiskLimits(max_net_delta={"NIFTY": 1000.0})
    orders = compute_hedge_orders(greeks, limits, {"NIFTY": snapshot.spot}, HedgeConfig())
    assert orders == []
