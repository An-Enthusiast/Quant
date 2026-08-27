"""Tests for the SVI surface calibration and no-arbitrage checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import svi_surface as svis


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
