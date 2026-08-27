"""Synthetic multi-tick option-chain generator for the backtester.

No live NSE tick history is available yet (see
docs/WHITEPAPER.md rollout notes -- Phase 1 fixture ingestion currently
uses a single static snapshot, data/sample_data/generate_fixtures.py). The
event-driven backtester needs a *chronological sequence* of snapshots to
replay, so this module generates one: a GBM spot path drives this
project's own Black-Scholes pricer (core/pricer_bindings.py) through a
fixed skew shape each tick, with per-contract open interest and cumulative
volume evolving as their own small random walks (not resampled i.i.d. each
tick, since the feature-engineering layer, alpha/features.py, depends on
realistic monotonic-ish OI/volume dynamics).

This is explicitly a synthetic data source, not a claim about real NSE
market behavior -- see the "Backtest Results" section of
docs/WHITEPAPER.md for how the resulting P&L/Sharpe numbers should (and
should not) be interpreted.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np

from core.option_chain import OptionChainSnapshot, OptionContract, OptionType
from core.pricer_bindings import price as bs_price


def _skewed_iv(k: float, base_iv: float, skew: float, convexity: float) -> float:
    return max(base_iv + skew * k + convexity * k * k, 0.03)


def generate_tick_series(
    symbol: str,
    spot0: float,
    strike_step: float,
    base_iv: float,
    expiry: date,
    n_ticks: int = 180,
    tick_interval_sec: float = 5.0,
    annual_vol: float = 0.15,
    n_strikes_each_side: int = 12,
    r: float = 0.065,
    q: float = 0.065,
    start_time: datetime | None = None,
    seed: int = 7,
) -> list[OptionChainSnapshot]:
    """Generates `n_ticks` chronologically ordered `OptionChainSnapshot`s for
    one underlying/expiry, spaced `tick_interval_sec` apart.
    """
    rng = np.random.default_rng(seed)
    start_time = start_time or datetime.now()
    dt_years = tick_interval_sec / (365.0 * 24 * 3600)

    atm_strike = round(spot0 / strike_step) * strike_step
    strikes = [atm_strike + i * strike_step for i in range(-n_strikes_each_side, n_strikes_each_side + 1)]
    strikes = [s for s in strikes if s > 0]

    # Per-contract running state (OI/volume evolve as their own random
    # walks across ticks rather than being resampled independently).
    state: dict[tuple[float, OptionType], dict[str, float]] = {}
    for strike in strikes:
        k0 = math.log(strike / spot0)
        decay = math.exp(-4.0 * k0 * k0)
        for opt_type in (OptionType.CALL, OptionType.PUT):
            state[(strike, opt_type)] = {
                "oi": max(rng.normal(150_000, 20_000), 500.0) * decay,
                "volume": 0.0,
            }

    snapshots: list[OptionChainSnapshot] = []
    spot = spot0
    for i in range(n_ticks):
        ts = start_time + timedelta(seconds=i * tick_interval_sec)
        spot *= math.exp(-0.5 * annual_vol**2 * dt_years + annual_vol * math.sqrt(dt_years) * rng.normal())

        days_to_expiry = (expiry - ts.date()).days
        T = max(days_to_expiry, 0) / 365.0 + max(0.0, (16 - ts.hour) / (365.0 * 24.0))
        T = max(T, 1.0 / (365.0 * 24.0 * 12.0))  # floor to avoid a literal zero-vega tick
        forward = spot * math.exp((r - q) * T)

        contracts: list[OptionContract] = []
        for strike in strikes:
            k = math.log(strike / forward)
            iv = _skewed_iv(k, base_iv=base_iv, skew=-0.5, convexity=1.8)
            for opt_type, is_call in ((OptionType.CALL, True), (OptionType.PUT, False)):
                st = state[(strike, opt_type)]
                theo = max(bs_price(spot, strike, T, r, q, iv, is_call), 0.05)
                half_spread = max(theo * 0.015, 0.05)
                bid = round(max(theo - half_spread, 0.0), 2)
                ask = round(theo + half_spread, 2)

                decay = math.exp(-4.0 * k * k)
                vol_increment = max(rng.normal(40.0, 20.0) * decay, 0.0)
                st["volume"] += vol_increment
                oi_delta = rng.normal(0.0, max(st["oi"] * 0.01, 1.0))
                st["oi"] = max(st["oi"] + oi_delta, 0.0)

                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=expiry,
                        strike=strike,
                        option_type=opt_type,
                        ltp=round(theo, 2),
                        bid=bid,
                        bid_qty=int(max(rng.normal(300, 80), 1)),
                        ask=ask,
                        ask_qty=int(max(rng.normal(300, 80), 1)),
                        oi=int(st["oi"]),
                        change_in_oi=int(oi_delta),
                        volume=int(st["volume"]),
                        timestamp=ts,
                        iv=round(iv * 100, 2),
                    )
                )

        snapshots.append(OptionChainSnapshot(symbol=symbol, timestamp=ts, spot=spot, contracts=contracts))

    return snapshots
