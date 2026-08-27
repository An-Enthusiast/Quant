"""Tests for the Black-Scholes pricer/Greeks/IV solver -- both the compiled
C++ `qengine` backend (via core.pricer_bindings) and the pure-Numba
fallback (core.greeks_numba), which must agree with each other.
"""

from __future__ import annotations

import math

import pytest

from core import greeks_numba, pricer_bindings

S, K, T, R, Q, SIGMA = 25000.0, 24800.0, 20.0 / 365.0, 0.065, 0.065, 0.14


def test_put_call_parity_cpp_backend():
    call = pricer_bindings.price(S, K, T, R, Q, SIGMA, True)
    put = pricer_bindings.price(S, K, T, R, Q, SIGMA, False)
    lhs = call - put
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_put_call_parity_numba_backend():
    call = greeks_numba.bs_price_scalar(S, K, T, R, Q, SIGMA, True)
    put = greeks_numba.bs_price_scalar(S, K, T, R, Q, SIGMA, False)
    lhs = call - put
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_cpp_and_numba_backends_agree():
    for is_call in (True, False):
        cpp_price = pricer_bindings.qengine.bs_price(S, K, T, R, Q, SIGMA, is_call) if pricer_bindings.using_cpp_engine() else None
        numba_price = greeks_numba.bs_price_scalar(S, K, T, R, Q, SIGMA, is_call)
        if cpp_price is not None:
            assert cpp_price == pytest.approx(numba_price, abs=1e-9)


def test_iv_roundtrip_cpp_backend():
    price = pricer_bindings.price(S, K, T, R, Q, SIGMA, False)
    res = pricer_bindings.implied_vol(price, S, K, T, R, Q, False)
    assert res.converged
    assert res.iv == pytest.approx(SIGMA, abs=1e-4)


def test_iv_roundtrip_numba_backend():
    price = greeks_numba.bs_price_scalar(S, K, T, R, Q, SIGMA, False)
    iv = greeks_numba.implied_vol_scalar(price, S, K, T, R, Q, False)
    assert iv == pytest.approx(SIGMA, abs=1e-4)


def test_iv_rejects_price_outside_no_arbitrage_bounds():
    upper_bound = S * math.exp(-Q * T)
    res = pricer_bindings.implied_vol(upper_bound + 100.0, S, K, T, R, Q, True)
    assert not res.converged


def test_greeks_delta_bounds():
    g = pricer_bindings.greeks(S, K, T, R, Q, SIGMA, True)
    assert 0.0 <= g.delta <= math.exp(-Q * T)
    g_put = pricer_bindings.greeks(S, K, T, R, Q, SIGMA, False)
    assert -math.exp(-Q * T) <= g_put.delta <= 0.0


def test_gamma_positive_and_matches_between_call_and_put():
    g_call = pricer_bindings.greeks(S, K, T, R, Q, SIGMA, True)
    g_put = pricer_bindings.greeks(S, K, T, R, Q, SIGMA, False)
    assert g_call.gamma > 0
    assert g_call.gamma == pytest.approx(g_put.gamma, rel=1e-9)


def test_vectorized_greeks_matches_scalar():
    import numpy as np

    n = 5
    Sarr = np.full(n, S)
    Karr = np.linspace(K - 500, K + 500, n)
    Tarr = np.full(n, T)
    Rarr = np.full(n, R)
    Qarr = np.full(n, Q)
    Sigarr = np.full(n, SIGMA)
    is_call = np.array([True, False, True, False, True])

    delta, gamma, theta, vega, rho = greeks_numba.vectorized_greeks(Sarr, Karr, Tarr, Rarr, Qarr, Sigarr, is_call)
    for i in range(n):
        d, g, t, v, r = greeks_numba.bs_greeks_scalar(Sarr[i], Karr[i], Tarr[i], Rarr[i], Qarr[i], Sigarr[i], bool(is_call[i]))
        assert delta[i] == pytest.approx(d)
        assert gamma[i] == pytest.approx(g)
        assert theta[i] == pytest.approx(t)
        assert vega[i] == pytest.approx(v)
        assert rho[i] == pytest.approx(r)
