"""Master orchestration entrypoint for the NSE Options Market Making &
Non-Linear Risk Hedging Engine.

    python main.py --mode prototype [--fixture] [--symbols NIFTY,BANKNIFTY]
    python main.py --mode live --broker shoonya|upstox
    python main.py --mode backtest [--source synthetic|duckdb]

--mode prototype
    Phase 1 (zero-cost): polls `NSEPythonAdapter` (core/nse_python_adapter.py),
    writes each snapshot to DuckDB, fits the SVI vol surface, and prints the
    resulting quotes each cycle. `--fixture` uses the recorded sample chain
    instead of a live nseindia.com call (this environment's default -- see
    docs/WHITEPAPER.md).

--mode live
    Phase 2: selects the Shoonya or Upstox streaming adapter
    (core/shoonya_ws_adapter.py / core/upstox_protobuf_adapter.py). Both
    share the exact same `MarketDataInterface` contract as the prototype
    adapter, so no pricer/alpha/risk code changes when switching modes.
    Both currently raise a clear `NotImplementedError` on `connect()` until
    real broker credentials and the websocket handshake are wired in.

--mode backtest
    Delegates to the event-driven backtester (backtest/run_backtest.py);
    see that module for the synthetic-vs-duckdb data source rationale.
"""

from __future__ import annotations

import argparse
import logging
import time

logger = logging.getLogger(__name__)


def run_prototype(args: argparse.Namespace) -> None:
    from alpha.quote_engine import QuoteEngine
    from core.nse_python_adapter import NSEPythonAdapter
    from core.svi_surface import fit_surface_from_chain
    from data.duckdb_store import DuckDBStore
    from risk.portfolio import Portfolio

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    adapter = NSEPythonAdapter(poll_interval_sec=args.interval, use_fixture=args.fixture)
    adapter.connect()
    quote_engine = QuoteEngine()
    portfolio = Portfolio()  # flat/no-inventory in this standalone prototype loop

    polls_done = 0
    try:
        with DuckDBStore(args.db) as store:
            while args.max_polls == 0 or polls_done < args.max_polls:
                for symbol in symbols:
                    try:
                        snapshot = adapter.get_option_chain(symbol)
                    except Exception:
                        logger.exception("prototype mode: failed to fetch %s", symbol)
                        continue
                    store.insert_snapshot(snapshot)
                    surface = fit_surface_from_chain(snapshot)
                    quotes = quote_engine.quote_snapshot(snapshot, surface=surface, portfolio=portfolio)
                    sample = quotes[:3]
                    print(
                        f"[{snapshot.timestamp}] {symbol} spot={snapshot.spot:.2f} "
                        f"contracts={len(snapshot.contracts)} quotes={len(quotes)}"
                    )
                    for q in sample:
                        _, expiry, strike, opt_type = q.contract_key
                        print(f"    {strike}{opt_type.value} {expiry}: bid={q.bid:.2f} ask={q.ask:.2f}")

                polls_done += 1
                if args.max_polls != 0 and polls_done >= args.max_polls:
                    break
                time.sleep(args.interval)
    finally:
        adapter.disconnect()


def run_live(args: argparse.Namespace) -> None:
    if args.broker == "shoonya":
        from core.shoonya_ws_adapter import ShoonyaCredentials, ShoonyaWebSocketAdapter

        credentials = ShoonyaCredentials(
            user_id="", password="", totp_secret="", vendor_code="", api_key="", imei=""
        )
        adapter = ShoonyaWebSocketAdapter(credentials)
    else:
        from core.upstox_protobuf_adapter import UpstoxCredentials, UpstoxProtobufAdapter

        credentials = UpstoxCredentials(client_id="", client_secret="", redirect_uri="")
        adapter = UpstoxProtobufAdapter(credentials)

    print(f"live mode: connecting via {args.broker} adapter (Phase 2)...")
    adapter.connect()  # raises NotImplementedError with setup instructions until credentials/impl are wired in


def run_backtest(args: argparse.Namespace) -> None:
    from backtest.event_engine import BacktestConfig, BacktestEngine
    from backtest.performance import build_report
    from backtest.run_backtest import SECONDS_PER_TRADING_YEAR, _generate_synthetic_session, _snapshots_from_duckdb
    from data.duckdb_store import DuckDBStore
    from risk.risk_limits import RiskLimits

    if args.source == "synthetic":
        snapshots_by_symbol = _generate_synthetic_session(args.n_ticks, args.tick_interval_sec, args.db)
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        with DuckDBStore(args.db) as store:
            snapshots_by_symbol = {symbol: _snapshots_from_duckdb(store, symbol) for symbol in symbols}

    limits = RiskLimits(
        max_net_delta={s: args.max_net_delta for s in snapshots_by_symbol},
        max_net_gamma={s: None for s in snapshots_by_symbol},
        max_net_vega={s: None for s in snapshots_by_symbol},
    )
    engine = BacktestEngine(BacktestConfig(risk_limits=limits))
    result = engine.run(snapshots_by_symbol)

    periods_per_year = SECONDS_PER_TRADING_YEAR / max(args.tick_interval_sec, 1e-9)
    pnl_values = [v for _, v in result.pnl_curve]
    report = build_report(pnl_values, len(result.trade_log), result.net_delta_history, periods_per_year)
    print(f"Fills/hedges executed: {len(result.trade_log)}")
    print(report.summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="NSE Options Market Making & Non-Linear Risk Hedging Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["prototype", "live", "backtest"],
        required=True,
        help="prototype = nsepython polling (Phase 1); live = broker websocket streaming (Phase 2); "
        "backtest = event-driven replay",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    proto = parser.add_argument_group("prototype mode")
    proto.add_argument("--symbols", default="NIFTY,BANKNIFTY")
    proto.add_argument("--interval", type=float, default=5.0)
    proto.add_argument("--max-polls", type=int, default=1, help="0 = run forever")
    proto.add_argument("--fixture", action="store_true", help="use the recorded sample chain, not live nsepython")
    proto.add_argument("--db", default="data/db/quant.duckdb")

    live = parser.add_argument_group("live mode")
    live.add_argument("--broker", choices=["shoonya", "upstox"], default="shoonya")

    bt = parser.add_argument_group("backtest mode")
    bt.add_argument("--source", choices=["synthetic", "duckdb"], default="synthetic")
    bt.add_argument("--n-ticks", type=int, default=180)
    bt.add_argument("--tick-interval-sec", type=float, default=5.0)
    bt.add_argument("--max-net-delta", type=float, default=300.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.mode == "prototype":
        run_prototype(args)
    elif args.mode == "live":
        run_live(args)
    elif args.mode == "backtest":
        run_backtest(args)


if __name__ == "__main__":
    main()
