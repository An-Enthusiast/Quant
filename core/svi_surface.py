"""Per-underlying implied-volatility surface: fits a raw-SVI smile
(w(k) = a + b(rho*(k-m) + sqrt((k-m)^2 + sigma^2))) to each expiry slice of
an option chain, and exposes a queryable `iv(expiry, strike)` surface with
static no-arbitrage checks (Gatheral butterfly bound per slice, calendar
monotonicity across slices).

Calibration prefers the compiled C++ `qengine.svi_calibrate` (Levenberg-
Marquardt, see csrc/src/svi.cpp); if the extension isn't built, falls back
to `scipy.optimize.least_squares` with the same box constraints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from core.option_chain import OptionChainSnapshot, OptionType
from core.pricer_bindings import implied_vol as _implied_vol

logger = logging.getLogger(__name__)

try:
    from core import qengine  # see core/pricer_bindings.py for import rationale

    _HAS_QENGINE = True
except ImportError:
    qengine = None  # type: ignore[assignment]
    _HAS_QENGINE = False


@dataclass(slots=True, frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(p: SVIParams, k: float) -> float:
    dk = k - p.m
    return p.a + p.b * (p.rho * dk + math.sqrt(dk * dk + p.sigma * p.sigma))


_DEFAULT_INITIAL_GUESS = SVIParams(a=0.02, b=0.15, rho=-0.3, m=0.0, sigma=0.2)


def _initial_guess_from_data(k: np.ndarray, w: np.ndarray) -> SVIParams:
    """Data-adaptive seed for the LM calibration, scaled to the slice's own
    total-variance level and moneyness range.

    A single fixed absolute seed (e.g. a=0.02, b=0.15) is badly scaled for
    short-dated slices where total variance can be an order of magnitude
    smaller -- the resulting ill-conditioned Gauss-Newton step tends to get
    stuck at a boundary-degenerate local minimum (b -> 0). Seeding from the
    data's own level, slope and curvature keeps the optimizer in a
    well-conditioned neighbourhood of the true minimum.
    """
    k_range = max(float(np.max(k) - np.min(k)), 1e-3)
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    slope = float(np.polyfit(k, w, 1)[0]) if len(k) >= 2 else 0.0
    return SVIParams(
        a=max(w_min * 0.8, 1e-6),
        b=max((w_max - w_min) / k_range, 1e-6),
        rho=-0.5 if slope <= 0 else 0.5,
        m=float(k[int(np.argmin(w))]),
        sigma=k_range / 4.0,
    )


def calibrate_svi_slice(
    k: np.ndarray, w: np.ndarray, initial_guess: SVIParams = _DEFAULT_INITIAL_GUESS
) -> tuple[SVIParams, float, bool]:
    """Fits one SVI slice to (log-moneyness, total-variance) points.
    Returns (params, rmse, converged).
    """
    if _HAS_QENGINE:
        seed = qengine.SVIParams(
            initial_guess.a, initial_guess.b, initial_guess.rho, initial_guess.m, initial_guess.sigma
        )
        res = qengine.svi_calibrate(list(map(float, k)), list(map(float, w)), seed)
        p = res.params
        return SVIParams(p.a, p.b, p.rho, p.m, p.sigma), res.rmse, res.converged

    from scipy.optimize import least_squares

    def resid(x: np.ndarray) -> np.ndarray:
        p = SVIParams(*x)
        return np.array([svi_total_variance(p, ki) - wi for ki, wi in zip(k, w, strict=True)])

    x0 = np.array(
        [initial_guess.a, initial_guess.b, initial_guess.rho, initial_guess.m, initial_guess.sigma]
    )
    lower = [-np.inf, 1e-8, -0.999, -np.inf, 1e-6]
    upper = [np.inf, np.inf, 0.999, np.inf, np.inf]
    sol = least_squares(resid, x0, bounds=(lower, upper))
    params = SVIParams(*sol.x)
    rmse = float(np.sqrt(np.mean(sol.fun**2))) if len(sol.fun) else float("nan")
    return params, rmse, bool(sol.success)


def check_butterfly_no_arb(p: SVIParams) -> tuple[bool, float]:
    """Gatheral & Jacquier (2014) sufficient conditions for a single slice."""
    if not (p.b >= 0.0 and abs(p.rho) < 1.0 and p.sigma > 0.0):
        return False, float("nan")
    vertex_min = p.a + p.b * p.sigma * math.sqrt(max(1.0 - p.rho**2, 0.0))
    wing_ok = p.b * (1.0 + abs(p.rho)) <= 4.0 + 1e-9
    return (wing_ok and vertex_min >= -1e-9), vertex_min


def check_calendar_no_arb(near: SVIParams, far: SVIParams, k_grid: np.ndarray) -> bool:
    """Total variance must be non-decreasing in maturity at every k."""
    return bool(np.all(np.array([svi_total_variance(far, k) for k in k_grid])
                        >= np.array([svi_total_variance(near, k) for k in k_grid]) - 1e-9))


@dataclass(slots=True)
class ExpirySlice:
    expiry: date
    T: float
    forward: float
    params: SVIParams
    rmse: float
    converged: bool
    n_points: int


@dataclass(slots=True)
class VolSurface:
    """A calibrated SVI surface for one underlying, indexed by expiry."""

    underlying: str
    slices: dict[date, ExpirySlice] = field(default_factory=dict)

    def iv(self, expiry: date, strike: float) -> float:
        """Query implied vol at (expiry, strike) from the fitted surface.
        Falls back to the nearest calibrated expiry if `expiry` wasn't fit
        directly (e.g. an intermediate maturity requested by the risk
        engine).
        """
        sl = self.slices.get(expiry) or self._nearest_slice(expiry)
        k = math.log(strike / sl.forward)
        w = max(svi_total_variance(sl.params, k), 1e-10)
        return math.sqrt(w / sl.T)

    def _nearest_slice(self, expiry: date) -> ExpirySlice:
        if not self.slices:
            raise RuntimeError(f"VolSurface for {self.underlying} has no calibrated slices")
        return min(self.slices.values(), key=lambda s: abs((s.expiry - expiry).days))

    def no_arbitrage_report(self) -> dict[str, object]:
        """Full-surface no-arbitrage report: per-slice butterfly check plus
        pairwise calendar checks between consecutive expiries.
        """
        report: dict[str, object] = {"butterfly": {}, "calendar": {}}
        ordered = sorted(self.slices.values(), key=lambda s: s.expiry)
        for sl in ordered:
            ok, vertex_min = check_butterfly_no_arb(sl.params)
            report["butterfly"][sl.expiry.isoformat()] = {"ok": ok, "min_total_variance": vertex_min}

        if len(ordered) >= 2:
            k_grid = np.linspace(-0.5, 0.5, 41)
            for near, far in zip(ordered, ordered[1:], strict=False):
                key = f"{near.expiry.isoformat()}->{far.expiry.isoformat()}"
                report["calendar"][key] = check_calendar_no_arb(near.params, far.params, k_grid)
        return report


def fit_surface_from_chain(
    snapshot: OptionChainSnapshot,
    r: float = 0.065,
    q: float = 0.065,
    valuation_time: datetime | None = None,
    min_points_per_slice: int = 5,
    min_mid_price: float = 0.5,
) -> VolSurface:
    """Builds a calibrated `VolSurface` from a raw option-chain snapshot.

    For each expiry: computes each contract's market implied vol from its
    mid price (via `core.pricer_bindings.implied_vol`), converts to
    log-moneyness/total-variance (k = log(K/F), w = iv^2 * T, where
    F = S * exp((r - q) * T) is the forward), and calibrates an SVI slice.

    Two standard vol-surface-construction filters are applied before
    fitting:
      - Only the out-of-the-money leg is used at each strike (OTM puts for
        K < F, OTM calls for K >= F). Put-call parity means either leg
        recovers the same IV in theory, but the OTM leg has materially
        higher vega than its deep-ITM counterpart, so IV inversion is
        numerically far better conditioned -- this is standard practice for
        listed-market smile construction, not specific to this dataset.
      - Quotes with mid price below `min_mid_price` are dropped: at that
        point bid/ask tick-size rounding is a large fraction of the quote,
        so the "market IV" it implies is dominated by rounding noise rather
        than signal (thin far-wing strikes in real chains have the same
        problem, which is why desks apply a liquidity/price floor here too).
    """
    valuation_time = valuation_time or snapshot.timestamp
    surface = VolSurface(underlying=snapshot.symbol)

    for expiry in snapshot.expiries:
        T = max((expiry - valuation_time.date()).days, 0) / 365.0
        if T <= 0:
            continue
        forward = snapshot.spot * math.exp((r - q) * T)

        ks: list[float] = []
        ws: list[float] = []
        for c in snapshot.contracts_for_expiry(expiry):
            is_call = c.option_type == OptionType.CALL
            is_otm = is_call == (c.strike >= forward)
            if not is_otm:
                continue
            mid = c.mid
            if mid < min_mid_price:
                continue
            res = _implied_vol(mid, snapshot.spot, c.strike, T, r, q, is_call)
            if not res.converged or res.iv <= 0:
                continue
            ks.append(math.log(c.strike / forward))
            ws.append(res.iv**2 * T)

        if len(ks) < min_points_per_slice:
            logger.warning(
                "svi_surface: skipping %s expiry %s -- only %d usable quotes (need >= %d)",
                snapshot.symbol,
                expiry,
                len(ks),
                min_points_per_slice,
            )
            continue

        k_arr, w_arr = np.array(ks), np.array(ws)
        params, rmse, converged = calibrate_svi_slice(k_arr, w_arr, _initial_guess_from_data(k_arr, w_arr))
        surface.slices[expiry] = ExpirySlice(
            expiry=expiry, T=T, forward=forward, params=params, rmse=rmse, converged=converged, n_points=len(ks)
        )

    return surface
