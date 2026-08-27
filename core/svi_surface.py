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


def pair_k_grid(k_min_a: float, k_max_a: float, k_min_b: float, k_max_b: float, n: int = 41) -> np.ndarray:
    """Grid spanning the union of two slices' *observed* log-moneyness
    ranges (with no extra margin) -- used both to check and to enforce
    calendar no-arbitrage between an adjacent pair of expiries.

    Deliberately not a fixed universal grid (e.g. a hardcoded +/-50%
    moneyness band): real strikes only span a bounded, day-specific range
    (a handful of percent for near-dated NIFTY weeklies, wider for far
    monthlies), and evaluating -- let alone enforcing -- calendar
    consistency far outside that range means extrapolating a 5-parameter
    curve into a region with no data to anchor it. Early testing against
    the real Bhavcopy archive with a fixed +/-50% grid did exactly this:
    the optimizer chased the (physically meaningless, way-out-of-domain)
    penalty by adopting extreme wing slopes, which *increased* measured
    calendar violations (78% -> 91%) and introduced new butterfly
    violations that weren't there before. Restricting to the pair's own
    traded domain is the same "domain bounds" principle already used for
    the base per-slice fit (see csrc/src/svi.cpp::DomainBounds), applied
    to the calendar constraint instead of just (m, sigma).
    """
    return np.linspace(min(k_min_a, k_min_b), max(k_max_a, k_max_b), n)


def _fit_with_calendar_penalty(
    k: np.ndarray, w: np.ndarray, x0: np.ndarray, bounds: tuple, floor_w_grid: np.ndarray, k_grid: np.ndarray,
    penalty_weight: float,
):
    from scipy.optimize import least_squares

    def resid(x: np.ndarray) -> np.ndarray:
        p = SVIParams(*x)
        market_resid = np.array([svi_total_variance(p, ki) - wi for ki, wi in zip(k, w, strict=True)])
        new_w_grid = np.array([svi_total_variance(p, kg) for kg in k_grid])
        violation = np.maximum(floor_w_grid - new_w_grid, 0.0)  # 0 when compliant, positive when violating
        return np.concatenate([market_resid, penalty_weight * violation])

    return least_squares(resid, x0, bounds=bounds)


def calibrate_svi_slice_with_calendar_floor(
    k: np.ndarray,
    w: np.ndarray,
    initial_guess: SVIParams,
    floor_params: SVIParams,
    k_grid: np.ndarray,
    penalty_weight: float = 50.0,
    max_rounds: int = 5,
    weight_growth: float = 10.0,
) -> tuple[SVIParams, float, bool]:
    """Fits one SVI slice like `calibrate_svi_slice`, with an added soft
    penalty enforcing `w(k) >= floor_params`'s `w(k)` (the immediately
    preceding, already-fitted, shorter-maturity slice) across `k_grid`
    (see `pair_k_grid` -- pass the pair's actual observed-data union, not
    an arbitrary wide domain) -- i.e. calendar no-arbitrage between this
    slice and its predecessor.

    Calendar monotonicity is transitive: if slice 2's total variance
    dominates slice 1's pointwise, and slice 3's dominates slice 2's, then
    slice 3's also dominates slice 1's. That means enforcing only
    *adjacent* pairs, fitted in increasing-maturity order, is sufficient
    to make the entire surface calendar-consistent within each pair's
    observed domain -- a fully joint optimization across every expiry
    simultaneously isn't needed. This is the sequential-constrained
    approach `fit_surface_from_chain` uses.

    Domain-derived bounds on (m, sigma) -- the same principle as
    csrc/src/svi.cpp::DomainBounds for the base fit -- are applied here
    too, scoped to the union of the market data and the penalty grid:
    without them the optimizer has nothing stopping it from reaching an
    extreme, poorly-identified parameterization while chasing the penalty
    (this was the root cause of an earlier regression: a single fixed
    weight against an unbounded fit made violations *worse*, not better).

    Escalating penalty weight, not a single fixed one. A moderate weight
    fully resolves an easy violation in one shot; a harder one (more
    tension between fitting the slice's own market quotes and dominating
    the floor) needs much more before the optimizer actually prioritizes
    it over market fit -- empirically, by roughly 2-4 orders of magnitude
    more for the hardest real cases found in the Bhavcopy archive. Rather
    than pick one weight large enough for the hardest case up front (which
    would needlessly distort the many easy cases), each round re-fits
    (warm-started from the previous round's result) with `weight_growth`x
    the penalty, stopping as soon as the pair is calendar-consistent on
    `k_grid`, up to `max_rounds`. If still violating after `max_rounds`,
    the last (highest-weight, closest) attempt is returned -- some
    genuine tension between market fit and calendar consistency will
    occasionally remain unresolved at any finite weight (see
    `fit_surface_from_chain`'s post-refit warning), which is exactly why
    `VolSurface.no_arbitrage_report()` remains a real, always-run check
    rather than an assumption this function guarantees.

    Uses `scipy.optimize.least_squares` regardless of whether the compiled
    C++ engine is available: the market-fit residual and the penalty
    residual both need to enter the *same* least-squares objective, and
    `svi_total_variance` (pure Python) is already the shared primitive the
    scipy fallback path in `calibrate_svi_slice` uses -- extending that
    path is far lower-risk than adding a second, differently-shaped
    objective to the hand-rolled C++ Levenberg-Marquardt loop for a case
    (recalibrating a handful of expiries per snapshot) that has no
    latency pressure to begin with (18 microseconds per unconstrained fit
    already, see docs/WHITEPAPER.md).
    """
    floor_w_grid = np.array([svi_total_variance(floor_params, kg) for kg in k_grid])

    domain_k_min = min(float(np.min(k)), float(np.min(k_grid)))
    domain_k_max = max(float(np.max(k)), float(np.max(k_grid)))
    domain_range = max(domain_k_max - domain_k_min, 1e-3)
    lower = [-np.inf, 1e-8, -0.999, domain_k_min - domain_range, 1e-6]
    upper = [np.inf, np.inf, 0.999, domain_k_max + domain_range, 2.0 * domain_range]

    x0 = np.clip(
        [initial_guess.a, initial_guess.b, initial_guess.rho, initial_guess.m, initial_guess.sigma], lower, upper
    )
    weight = penalty_weight
    sol = None
    for _round in range(max_rounds):
        sol = _fit_with_calendar_penalty(k, w, x0, (lower, upper), floor_w_grid, k_grid, weight)
        params = SVIParams(*sol.x)
        if check_calendar_no_arb(floor_params, params, k_grid):
            break
        x0 = sol.x
        weight *= weight_growth

    params = SVIParams(*sol.x)
    market_resid = sol.fun[: len(k)]
    rmse = float(np.sqrt(np.mean(market_resid**2))) if len(market_resid) else float("nan")
    return params, rmse, bool(sol.success)


@dataclass(slots=True)
class ExpirySlice:
    expiry: date
    T: float
    forward: float
    params: SVIParams
    rmse: float
    converged: bool
    n_points: int
    k_min: float = 0.0  # observed log-moneyness range actually used to fit this
    k_max: float = 0.0  # slice -- see pair_k_grid for why this matters for calendar checks
    calendar_adjusted: bool = False  # True if the unconstrained fit violated calendar
    # no-arb against the previous (shorter-maturity) slice and had to be
    # refit with calibrate_svi_slice_with_calendar_floor.


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
            for near, far in zip(ordered, ordered[1:], strict=False):
                key = f"{near.expiry.isoformat()}->{far.expiry.isoformat()}"
                grid = pair_k_grid(near.k_min, near.k_max, far.k_min, far.k_max)
                report["calendar"][key] = check_calendar_no_arb(near.params, far.params, grid)
        return report


def fit_surface_from_chain(
    snapshot: OptionChainSnapshot,
    r: float = 0.065,
    q: float = 0.065,
    valuation_time: datetime | None = None,
    min_points_per_slice: int = 5,
    min_mid_price: float = 0.5,
    enforce_calendar_no_arb: bool = True,
) -> VolSurface:
    """Builds a calibrated `VolSurface` from a raw option-chain snapshot.

    For each expiry (fit in increasing-maturity order): computes each
    contract's market implied vol from its mid price (via
    `core.pricer_bindings.implied_vol`), converts to log-moneyness/total-
    variance (k = log(K/F), w = iv^2 * T, where F = S * exp((r - q) * T)
    is the forward), and calibrates an SVI slice.

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

    Joint (sequential) calendar calibration
    -----------------------------------------
    Independent per-slice fitting has no structural guarantee against
    calendar arbitrage (w must be non-decreasing in maturity at every k) --
    measured against the full real Bhavcopy archive, 78-83% of expiry-pairs
    violated it (see docs/WHITEPAPER.md). When `enforce_calendar_no_arb`
    is True (the default): each slice after the first is fit unconstrained
    first (cheap, and often already compliant), then checked against the
    immediately preceding *successfully fitted* slice (a skipped expiry
    doesn't become the floor -- the last one that actually fit does); a
    violation triggers a refit via `calibrate_svi_slice_with_calendar_floor`,
    seeded from the unconstrained result. Enforcing only adjacent pairs in
    maturity order is sufficient for the whole surface to be calendar-
    consistent (the property is transitive -- see that function's
    docstring), so this stays a sequence of small, cheap 5-parameter fits
    rather than one large joint optimization across every expiry. Set to
    False to get the old independent-per-slice behavior (e.g. for direct
    comparison/diagnostics).
    """
    valuation_time = valuation_time or snapshot.timestamp
    surface = VolSurface(underlying=snapshot.symbol)
    prev_slice: ExpirySlice | None = None

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
        k_min, k_max = float(np.min(k_arr)), float(np.max(k_arr))
        initial_guess = _initial_guess_from_data(k_arr, w_arr)
        params, rmse, converged = calibrate_svi_slice(k_arr, w_arr, initial_guess)
        calendar_adjusted = False

        if enforce_calendar_no_arb and prev_slice is not None:
            grid = pair_k_grid(prev_slice.k_min, prev_slice.k_max, k_min, k_max)
            if not check_calendar_no_arb(prev_slice.params, params, grid):
                params, rmse, converged = calibrate_svi_slice_with_calendar_floor(
                    k_arr, w_arr, initial_guess=params, floor_params=prev_slice.params, k_grid=grid
                )
                calendar_adjusted = True
                if not check_calendar_no_arb(prev_slice.params, params, grid):
                    logger.warning(
                        "svi_surface: %s expiry %s still violates calendar no-arb against %s after "
                        "the constrained refit -- the market data may not admit a calendar-consistent "
                        "fit at the current penalty_weight",
                        snapshot.symbol,
                        expiry,
                        prev_slice.expiry,
                    )

        prev_slice = ExpirySlice(
            expiry=expiry,
            T=T,
            forward=forward,
            params=params,
            rmse=rmse,
            converged=converged,
            n_points=len(ks),
            k_min=k_min,
            k_max=k_max,
            calendar_adjusted=calendar_adjusted,
        )
        surface.slices[expiry] = prev_slice

    return surface
