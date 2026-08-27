"""Backtest performance analytics: Sharpe, Sortino, max drawdown, and
inventory-risk summary statistics computed from a mark-to-market P&L curve
and the desk's net-delta history over the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class PerformanceReport:
    total_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    n_periods: int
    n_trades: int
    avg_abs_net_delta: float
    max_abs_net_delta: float

    def summary(self) -> str:
        return (
            f"Total P&L: {self.total_pnl:,.2f}\n"
            f"Sharpe ratio (annualized): {self.sharpe_ratio:.3f}\n"
            f"Sortino ratio (annualized): {self.sortino_ratio:.3f}\n"
            f"Max drawdown: {self.max_drawdown:,.2f}\n"
            f"Periods: {self.n_periods}, Trades: {self.n_trades}\n"
            f"Avg |net delta|: {self.avg_abs_net_delta:,.2f}, Max |net delta|: {self.max_abs_net_delta:,.2f}"
        )


def _period_pnl(pnl_curve: np.ndarray) -> np.ndarray:
    return np.diff(pnl_curve)


def sharpe_ratio(period_pnl: np.ndarray, periods_per_year: float) -> float:
    if len(period_pnl) < 2 or np.std(period_pnl) == 0:
        return 0.0
    return float(np.mean(period_pnl) / np.std(period_pnl) * np.sqrt(periods_per_year))


def sortino_ratio(period_pnl: np.ndarray, periods_per_year: float) -> float:
    downside = period_pnl[period_pnl < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(period_pnl) / np.std(downside) * np.sqrt(periods_per_year))


def max_drawdown(pnl_curve: np.ndarray) -> float:
    if len(pnl_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(pnl_curve)
    return float(np.min(pnl_curve - running_max))


def build_report(
    pnl_curve: list[float], n_trades: int, net_delta_history: list[float], periods_per_year: float
) -> PerformanceReport:
    arr = np.array(pnl_curve, dtype=np.float64)
    period_pnl = _period_pnl(arr) if len(arr) else np.array([])
    delta_arr = np.array(net_delta_history, dtype=np.float64) if net_delta_history else np.array([0.0])

    return PerformanceReport(
        total_pnl=float(arr[-1] - arr[0]) if len(arr) else 0.0,
        sharpe_ratio=sharpe_ratio(period_pnl, periods_per_year),
        sortino_ratio=sortino_ratio(period_pnl, periods_per_year),
        max_drawdown=max_drawdown(arr),
        n_periods=len(arr),
        n_trades=n_trades,
        avg_abs_net_delta=float(np.mean(np.abs(delta_arr))),
        max_abs_net_delta=float(np.max(np.abs(delta_arr))),
    )
