"""Tests for the SVI surface calibration and no-arbitrage checks."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from core import svi_surface as svis
from core.option_chain import OptionChainSnapshot, OptionContract, OptionType
from core.pricer_bindings import price as bs_price


def test_svi_calibration_recovers_seeded_params():
    truth = svis.SVIParams(a=0.02, b=0.15, rho=-0.3, m=0.0, sigma=0.2)
    k = np.linspace(-0.5, 0.5, 21)
    w = np.array([svis.svi_total_variance(truth, ki) for ki in k])

    guess = svis._initial_guess_from_data(k, w)
    fitted, rmse, converged = svis.calibrate_svi_slice(k, w, guess)

    assert converged
    assert rmse < 1e-4
    for ki in k:
        assert svis.svi_total_variance(fitted, ki) == pytest.approx(svis.svi_total_variance(truth, ki), abs=1e-3)


def test_svi_calibration_with_data_adaptive_guess_on_realistic_smile():
    truth = svis.SVIParams(a=0.0003, b=0.002, rho=-0.9, m=-0.05, sigma=0.15)
    k = np.linspace(-0.09, 0.06, 34)
    w = np.array([svis.svi_total_variance(truth, ki) for ki in k])

    guess = svis._initial_guess_from_data(k, w)
    fitted, rmse, converged = svis.calibrate_svi_slice(k, w, guess)

    assert converged
    assert rmse < 1e-5


def test_butterfly_no_arb_check_passes_for_valid_params():
    p = svis.SVIParams(a=0.02, b=0.15, rho=-0.3, m=0.0, sigma=0.2)
    ok, vertex_min = svis.check_butterfly_no_arb(p)
    assert ok
    assert vertex_min >= 0


def test_butterfly_no_arb_check_fails_for_negative_vertex():
    p = svis.SVIParams(a=-1.0, b=0.15, rho=-0.3, m=0.0, sigma=0.2)
    ok, _ = svis.check_butterfly_no_arb(p)
    assert not ok


def test_butterfly_no_arb_check_fails_for_steep_wing():
    p = svis.SVIParams(a=0.02, b=10.0, rho=-0.5, m=0.0, sigma=0.2)  # b*(1+|rho|) way over 4
    ok, _ = svis.check_butterfly_no_arb(p)
    assert not ok


def test_calendar_no_arb_check():
    near = svis.SVIParams(a=0.01, b=0.10, rho=-0.2, m=0.0, sigma=0.15)
    far = svis.SVIParams(a=0.02, b=0.12, rho=-0.2, m=0.0, sigma=0.18)
    k_grid = np.linspace(-0.3, 0.3, 21)
    assert svis.check_calendar_no_arb(near, far, k_grid)
    assert not svis.check_calendar_no_arb(far, near, k_grid)


def test_fit_surface_from_chain_and_no_arbitrage_report():
    from core.nse_python_adapter import NSEPythonAdapter

    adapter = NSEPythonAdapter(use_fixture=True)
    adapter.connect()
    snapshot = adapter.get_option_chain("NIFTY")

    surface = svis.fit_surface_from_chain(snapshot)
    assert len(surface.slices) >= 2

    e0 = snapshot.expiries[0]
    atm_iv = surface.iv(e0, snapshot.spot)
    assert 0.03 < atm_iv < 1.0

    # Negative skew: OTM puts should show higher IV than OTM calls
    put_wing_iv = surface.iv(e0, snapshot.spot * 0.92)
    call_wing_iv = surface.iv(e0, snapshot.spot * 1.08)
    assert put_wing_iv > atm_iv > call_wing_iv

    report = surface.no_arbitrage_report()
    assert "butterfly" in report and "calendar" in report
    assert all(v["ok"] for v in report["butterfly"].values())


def _flat_smile_snapshot(
    symbol: str,
    spot: float,
    strike_step: float,
    valuation_dt: datetime,
    expiry_ivs: list[tuple[int, float]],
    r: float = 0.065,
    q: float = 0.065,
    n_strikes: int = 15,
) -> OptionChainSnapshot:
    """A synthetic snapshot with a flat (non-skewed) IV per expiry, from
    `expiry_ivs`: (days_to_expiry, iv) pairs. Used to deliberately
    construct a calendar violation by giving a near expiry a higher IV
    than a farther one -- w = iv^2 * T can then be *lower* for the longer
    maturity, exactly the arbitrage the calendar check exists to catch.
    """
    contracts = []
    atm_strike = round(spot / strike_step) * strike_step
    for days, iv in expiry_ivs:
        expiry = valuation_dt.date() + timedelta(days=days)
        T = days / 365.0
        forward = spot * math.exp((r - q) * T)
        for i in range(-n_strikes, n_strikes + 1):
            strike = atm_strike + i * strike_step
            if strike <= 0:
                continue
            for opt_type, is_call in ((OptionType.CALL, True), (OptionType.PUT, False)):
                theo = max(bs_price(spot, strike, T, r, q, iv, is_call), 0.05)
                half_spread = max(theo * 0.01, 0.05)
                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=expiry,
                        strike=strike,
                        option_type=opt_type,
                        ltp=round(theo, 2),
                        bid=round(theo - half_spread, 2),
                        bid_qty=100,
                        ask=round(theo + half_spread, 2),
                        ask_qty=100,
                        oi=10000,
                        change_in_oi=0,
                        volume=1000,
                        timestamp=valuation_dt,
                        iv=None,
                    )
                )
    return OptionChainSnapshot(symbol=symbol, timestamp=valuation_dt, spot=spot, contracts=contracts)


def test_joint_calendar_calibration_fixes_a_real_violation():
    """Regression test for the finding on the real Bhavcopy archive: 78-83%
    of expiry-pairs violated calendar no-arb under independent per-slice
    fitting; joint (sequential, calendar-floor-constrained) calibration
    brought that down to 1.4%/3.1%. This reproduces one such violation
    deterministically (a near expiry with a much higher flat IV than the
    far expiry) and checks the fix actually fixes it.
    """
    valuation_dt = datetime(2026, 8, 1, 15, 30)
    snapshot = _flat_smile_snapshot(
        "TEST", spot=25000.0, strike_step=50.0, valuation_dt=valuation_dt, expiry_ivs=[(7, 0.30), (14, 0.15)]
    )

    unconstrained = svis.fit_surface_from_chain(snapshot, valuation_time=valuation_dt, enforce_calendar_no_arb=False)
    assert len(unconstrained.slices) == 2
    unconstrained_report = unconstrained.no_arbitrage_report()
    assert not all(unconstrained_report["calendar"].values())  # the deliberate violation shows up

    joint = svis.fit_surface_from_chain(snapshot, valuation_time=valuation_dt, enforce_calendar_no_arb=True)
    assert len(joint.slices) == 2
    joint_report = joint.no_arbitrage_report()
    assert all(joint_report["calendar"].values())  # joint calibration fixes it

    far_slice = sorted(joint.slices.values(), key=lambda s: s.expiry)[1]
    assert far_slice.calendar_adjusted


def test_calibrate_svi_slice_with_calendar_floor_dominates_and_fits_reasonably():
    floor_params = svis.SVIParams(a=0.0017, b=1e-8, rho=0.0, m=0.0, sigma=0.05)  # ~flat w=0.0017
    truth = svis.SVIParams(a=0.0009, b=1e-8, rho=0.0, m=0.0, sigma=0.05)  # ~flat w=0.0009, violates floor
    k = np.linspace(-0.08, 0.08, 21)
    w = np.array([svis.svi_total_variance(truth, ki) for ki in k])

    grid = svis.pair_k_grid(float(k.min()), float(k.max()), float(k.min()), float(k.max()))
    assert not svis.check_calendar_no_arb(floor_params, truth, grid)  # sanity: truth really violates

    fitted, rmse, converged = svis.calibrate_svi_slice_with_calendar_floor(
        k, w, initial_guess=truth, floor_params=floor_params, k_grid=grid
    )
    assert svis.check_calendar_no_arb(floor_params, fitted, grid)
    # Still a reasonable fit to its own market data, not a wild distortion --
    # the floor is only ~0.0008 above the true level, so a modest, not
    # extreme, RMSE increase is expected.
    assert rmse < 2e-3


def test_pair_k_grid_spans_the_union_of_both_ranges():
    grid = svis.pair_k_grid(-0.1, 0.05, -0.03, 0.12, n=11)
    assert grid.min() == pytest.approx(-0.1)
    assert grid.max() == pytest.approx(0.12)
    assert len(grid) == 11
