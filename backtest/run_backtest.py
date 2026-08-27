"""CLI: run the event-driven backtest and print a performance report.

No live/recorded NSE tick history is available yet (see
docs/WHITEPAPER.md), so `--source synthetic` (the default) generates a
GBM-driven synthetic multi-tick session per underlying
(backtest/synthetic_ticks.py) and ingests it into a dedicated DuckDB file
before replaying it -- exercising the full ingest -> DuckDB -> backtest
pipeline end to end, just with a documented synthetic origin rather than a
live feed. `--source duckdb` replays whatever real history has already been
ingested into an existing DuckDB (e.g. via `python -m data.ingest --mode
live`), once that's available.

Examples
--------
    python -m backtest.run_backtest --source synthetic --n-ticks 180
    python -m backtest.run_backtest --source duckdb --db data/db/quant.duckdb --symbol NIFTY
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from backtest.event_engine import BacktestConfig, BacktestEngine
from backtest.performance import build_report
from backtest.synthetic_ticks import generate_tick_series
from core.option_chain import OptionChainSnapshot
from data.duckdb_store import DuckDBStore
from risk.risk_limits import RiskLimits

logger = logging.getLogger(__name__)

SECONDS_PER_TRADING_YEAR = 252 * 6.25 * 3600  # NSE cash/derivatives session length


def _snapshots_from_duckdb(store: DuckDBStore, symbol: str) -> list[OptionChainSnapshot]:
    import pandas as pd

    from core.option_chain import OptionContract, OptionType

    df = store.query_range(symbol)
    snapshots: list[OptionChainSnapshot] = []
    for raw_ts, group in df.groupby("ts"):
        ts = pd.Timestamp(raw_ts).to_pydatetime()
        spot = float(group["underlying_spot"].iloc[0])
        contracts = [
            OptionContract(
                symbol=symbol,
                expiry=pd.Timestamp(row.expiry).date(),
                strike=float(row.strike),
                option_type=OptionType(row.option_type),
                ltp=float(row.ltp),
                bid=float(row.bid),
                bid_qty=int(row.bid_qty),
                ask=float(row.ask),
                ask_qty=int(row.ask_qty),
                oi=int(row.oi),
                change_in_oi=int(row.change_in_oi),
                volume=int(row.volume),
                timestamp=ts,
                iv=float(row.exchange_iv) if row.exchange_iv is not None else None,
            )
            for row in group.itertuples()
        ]
        snapshots.append(OptionChainSnapshot(symbol=symbol, timestamp=ts, spot=spot, contracts=contracts))
    return snapshots


def _generate_synthetic_session(n_ticks: int, tick_interval_sec: float, db_path: str) -> dict[str, list[OptionChainSnapshot]]:
    start_time = datetime.now()
    weekly_expiry = start_time.date().fromordinal(start_time.date().toordinal() + 7)

    series = {
        "NIFTY": generate_tick_series(
            "NIFTY", spot0=25000.0, strike_step=50.0, base_iv=0.13, expiry=weekly_expiry,
            n_ticks=n_ticks, tick_interval_sec=tick_interval_sec, start_time=start_time, seed=7,
        ),
        "BANKNIFTY": generate_tick_series(
            "BANKNIFTY", spot0=52000.0, strike_step=100.0, base_iv=0.16, expiry=weekly_expiry,
            n_ticks=n_ticks, tick_interval_sec=tick_interval_sec, start_time=start_time, seed=11,
        ),
    }

    with DuckDBStore(db_path) as store:
        for symbol, snaps in series.items():
            for snap in snaps:
                store.insert_snapshot(snap)
        logger.info("Synthetic session ingested into %s (%d total rows)", db_path, store.row_count())

    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the event-driven options market-making backtest")
    parser.add_argument("--source", choices=["synthetic", "duckdb"], default="synthetic")
    parser.add_argument("--db", default="data/db/backtest_synthetic.duckdb")
    parser.add_argument("--symbol", default="NIFTY", help="Only used with --source duckdb")
    parser.add_argument("--n-ticks", type=int, default=180)
    parser.add_argument("--tick-interval-sec", type=float, default=5.0)
    parser.add_argument("--max-net-delta", type=float, default=300.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.source == "synthetic":
        snapshots_by_symbol = _generate_synthetic_session(args.n_ticks, args.tick_interval_sec, args.db)
    else:
        with DuckDBStore(args.db) as store:
            snapshots_by_symbol = {args.symbol: _snapshots_from_duckdb(store, args.symbol)}

    for symbol, snaps in snapshots_by_symbol.items():
        print(f"{symbol}: {len(snaps)} ticks loaded")

    limits = RiskLimits(
        max_net_delta={s: args.max_net_delta for s in snapshots_by_symbol},
        max_net_gamma={s: None for s in snapshots_by_symbol},
        max_net_vega={s: None for s in snapshots_by_symbol},
    )
    config = BacktestConfig(risk_limits=limits)
    engine = BacktestEngine(config)
    result = engine.run(snapshots_by_symbol)

    periods_per_year = SECONDS_PER_TRADING_YEAR / max(args.tick_interval_sec, 1e-9)
    pnl_values = [v for _, v in result.pnl_curve]
    report = build_report(pnl_values, len(result.trade_log), result.net_delta_history, periods_per_year)

    print()
    print(f"Fills/hedges executed: {len(result.trade_log)}")
    print(report.summary())


if __name__ == "__main__":
    main()
