"""Unified pricer facade: prefers the compiled C++ `qengine` pybind11
extension (csrc/) for scalar pricing/Greeks/IV, and always uses the
Numba-vectorized engine (core/greeks_numba.py) for whole-chain batch
Greeks. If `qengine` isn't built, scalar calls transparently fall back to
the same Numba implementation -- Module A's "expose via pybind11 or Numba"
requirement is satisfied either way, and no caller needs to know which
backend served a given call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core import greeks_numba

logger = logging.getLogger(__name__)

try:
    # Compiled from csrc/ via CMake (see csrc/CMakeLists.txt), which places
    # the extension module inside this package directory as core/qengine*.so
    # -- imported here as a submodule of `core` rather than a top-level
    # `import qengine` so it resolves regardless of the caller's cwd/sys.path.
    from core import qengine

    _HAS_QENGINE = True
    logger.info("pricer_bindings: using compiled C++ qengine backend")
except ImportError:
    qengine = None  # type: ignore[assignment]
    _HAS_QENGINE = False
    logger.warning(
        "pricer_bindings: compiled qengine extension not found; falling back to the "
        "Numba implementation (core/greeks_numba.py). Build csrc/ for the low-latency "
        "C++ path -- see deployment/Dockerfile or the top-level setup guide."
    )


@dataclass(slots=True, frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(slots=True, frozen=True)
class IVResult:
    iv: float
    converged: bool


def using_cpp_engine() -> bool:
    return _HAS_QENGINE


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    if _HAS_QENGINE:
        return qengine.bs_price(S, K, T, r, q, sigma, is_call)
    return greeks_numba.bs_price_scalar(S, K, T, r, q, sigma, is_call)


def greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> Greeks:
    if _HAS_QENGINE:
        g = qengine.bs_greeks(S, K, T, r, q, sigma, is_call)
        return Greeks(g.delta, g.gamma, g.theta, g.vega, g.rho)
    d, g, t, v, rh = greeks_numba.bs_greeks_scalar(S, K, T, r, q, sigma, is_call)
    return Greeks(d, g, t, v, rh)


def implied_vol(
    price_: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    initial_guess: float = 0.3,
) -> IVResult:
    if _HAS_QENGINE:
        res = qengine.implied_vol(price_, S, K, T, r, q, is_call, initial_guess)
        return IVResult(res.iv, res.converged)
    iv = greeks_numba.implied_vol_scalar(price_, S, K, T, r, q, is_call, initial_guess)
    import math

    return IVResult(iv, not math.isnan(iv))
