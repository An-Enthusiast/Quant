"""Numba-accelerated Black-Scholes pricing, Greeks and implied-vol solver.

Serves two purposes:
  1. The vectorized Greeks engine of Module A: `vectorized_greeks` prices an
     entire option chain (every live Nifty/BankNifty strike/expiry) in one
     parallel `@njit` call instead of a Python loop.
  2. The pure-Python/Numba fallback path for `core/pricer_bindings.py` when
     the compiled C++ `qengine` extension (csrc/) hasn't been built --
     scalar functions here mirror `csrc/src/bs_pricer.cpp` formula-for-formula
     (same Newton-Raphson-with-bisection-fallback IV solver) so results are
     consistent regardless of which backend is active.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_INV_SQRT_2PI = 0.3989422804014327
_SQRT2 = math.sqrt(2.0)


@njit(cache=True)
def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT2)


@njit(cache=True)
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) * _INV_SQRT_2PI


@njit(cache=True)
def bs_price_scalar(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    if T <= 0.0 or sigma <= 0.0:
        fwd_diff = S * math.exp(-q * max(T, 0.0)) - K * math.exp(-r * max(T, 0.0))
        return max(fwd_diff, 0.0) if is_call else max(-fwd_diff, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if is_call:
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)


@njit(cache=True)
def bs_greeks_scalar(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool):
    """Returns (delta, gamma, theta, vega, rho)."""
    if T <= 0.0 or sigma <= 0.0:
        fwd_diff = S * math.exp(-q * max(T, 0.0)) - K * math.exp(-r * max(T, 0.0))
        itm = (fwd_diff > 0.0) if is_call else (fwd_diff < 0.0)
        delta = (1.0 if is_call else -1.0) if itm else 0.0
        return delta, 0.0, 0.0, 0.0, 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    if is_call:
        delta = disc_q * _norm_cdf(d1)
        theta_annual = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * disc_r * _norm_cdf(
            d2
        ) + q * S * disc_q * _norm_cdf(d1)
        theta = theta_annual / 365.0
        rho = K * T * disc_r * _norm_cdf(d2) / 100.0
    else:
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta_annual = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * disc_r * _norm_cdf(
            -d2
        ) - q * S * disc_q * _norm_cdf(-d1)
        theta = theta_annual / 365.0
        rho = -K * T * disc_r * _norm_cdf(-d2) / 100.0

    return delta, gamma, theta, vega, rho


@njit(cache=True)
def implied_vol_scalar(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    initial_guess: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 100,
    vega_floor: float = 1e-8,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Safeguarded Newton-Raphson with bisection fallback; mirrors
    `qengine::implied_vol` in csrc/src/bs_pricer.cpp. Returns NaN if `price`
    is outside no-arbitrage bounds.
    """
    intrinsic = bs_price_scalar(S, K, T, r, q, 1e-6, is_call)
    upper_bound = S * math.exp(-q * T) if is_call else K * math.exp(-r * T)
    if T <= 0.0 or price < intrinsic - 1e-10 or price > upper_bound + 1e-10:
        return math.nan

    sigma = min(max(initial_guess, lo), hi)
    for i in range(max_iter):
        model_price = bs_price_scalar(S, K, T, r, q, sigma, is_call)
        diff = model_price - price
        if abs(diff) < tol:
            return sigma
        _, _, _, vega, _ = bs_greeks_scalar(S, K, T, r, q, sigma, is_call)
        if vega < vega_floor:
            blo, bhi = lo, hi
            f_lo = bs_price_scalar(S, K, T, r, q, blo, is_call) - price
            for _j in range(max_iter - i):
                mid = 0.5 * (blo + bhi)
                f_mid = bs_price_scalar(S, K, T, r, q, mid, is_call) - price
                if abs(f_mid) < tol:
                    return mid
                if (f_lo < 0.0) == (f_mid < 0.0):
                    blo = mid
                    f_lo = f_mid
                else:
                    bhi = mid
            return 0.5 * (blo + bhi)
        sigma = min(max(sigma - diff / vega, lo), hi)
    return sigma


@njit(cache=True, parallel=True)
def vectorized_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
):
    """Prices Delta/Gamma/Theta/Vega/Rho for an entire option chain at once.

    All inputs are 1D arrays of equal length (one row per contract);
    `is_call` is a boolean array. Runs across cores via `numba.prange` --
    on a multi-hundred-contract Nifty+BankNifty chain this is the
    difference between microseconds and milliseconds per full-chain
    Greeks refresh.
    """
    n = S.shape[0]
    delta = np.empty(n, dtype=np.float64)
    gamma = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    vega = np.empty(n, dtype=np.float64)
    rho = np.empty(n, dtype=np.float64)
    for i in prange(n):
        d, g, t, v, rh = bs_greeks_scalar(S[i], K[i], T[i], r[i], q[i], sigma[i], is_call[i])
        delta[i] = d
        gamma[i] = g
        theta[i] = t
        vega[i] = v
        rho[i] = rh
    return delta, gamma, theta, vega, rho


@njit(cache=True, parallel=True)
def vectorized_price(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, q: np.ndarray, sigma: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    n = S.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = bs_price_scalar(S[i], K[i], T[i], r[i], q[i], sigma[i], is_call[i])
    return out


@njit(cache=True, parallel=True)
def vectorized_implied_vol(
    price: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, q: np.ndarray, is_call: np.ndarray
) -> np.ndarray:
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = implied_vol_scalar(price[i], S[i], K[i], T[i], r[i], q[i], is_call[i])
    return out
