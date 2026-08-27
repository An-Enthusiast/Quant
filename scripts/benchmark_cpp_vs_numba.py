"""Benchmark: run the real SVI-fitting pipeline against the full Bhavcopy
archive under both pricing/calibration backends -- the compiled C++
`qengine` extension, and the pure-Python fallback (Numba for
price/Greeks/IV via core/greeks_numba.py, SciPy for SVI calibration since
Numba doesn't do nonlinear least-squares) -- and report wall-clock time
for each, plus a correctness sanity check that both backends produce
consistent fits.

Each backend runs in its own subprocess: `core.pricer_bindings` and
`core.svi_surface` decide which backend to use once, at first import
(`_HAS_QENGINE`), so comparing both within one process would need fragile
module-reload gymnastics. The fallback subprocess poisons
`sys.modules['core.qengine'] = None` before any project import, which
makes `from core import qengine` raise `ImportError` -- the same effect
as the extension never having been built, with no filesystem changes (no
need to move core/qengine*.so aside and remember to put it back).

Usage
-----
    python scripts/benchmark_cpp_vs_numba.py --db data/db/quant.duckdb
    python scripts/benchmark_cpp_vs_numba.py --symbols NIFTY

Internal (used by the subprocess dispatch above, not meant to be run
directly):
    python scripts/benchmark_cpp_vs_numba.py --_worker cpp --db ... --symbols ...
    python scripts/benchmark_cpp_vs_numba.py --_worker fallback --db ... --symbols ...
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _run_worker(db_path: str, symbols: list[str]) -> dict:
    """Runs fit_surface_from_chain across every ingested day for `symbols`
    and returns timing + correctness stats. Imports are deliberately local
    to this function: in the `fallback` subprocess, sys.modules['core.qengine']
    is poisoned to None *before* this (or anything under core/) is
    imported, so the backend selection in pricer_bindings/svi_surface picks
    up the fallback path.
    """
    import logging as _logging

    _logging.disable(_logging.WARNING)  # keep timing runs quiet; correctness already covered by tests/

    from backtest.run_backtest import _snapshots_from_duckdb
    from core import pricer_bindings
    from core.svi_surface import fit_surface_from_chain
    from data.duckdb_store import DuckDBStore

    per_symbol = {}
    total_start = time.perf_counter()

    with DuckDBStore(db_path) as store:
        for symbol in symbols:
            snapshots = _snapshots_from_duckdb(store, symbol)
            n_days = len(snapshots)
            fitted_expiries = 0
            converged = 0
            errors = 0
            rmse_sum = 0.0

            t0 = time.perf_counter()
            for snapshot in snapshots:
                try:
                    surface = fit_surface_from_chain(snapshot)
                except Exception:
                    errors += 1
                    continue
                for sl in surface.slices.values():
                    fitted_expiries += 1
                    rmse_sum += sl.rmse
                    if sl.converged:
                        converged += 1
            elapsed = time.perf_counter() - t0

            per_symbol[symbol] = {
                "n_days": n_days,
                "elapsed_sec": elapsed,
                "fitted_expiries": fitted_expiries,
                "converged": converged,
                "errors": errors,
                "mean_rmse": rmse_sum / fitted_expiries if fitted_expiries else None,
            }

    total_elapsed = time.perf_counter() - total_start
    return {
        "using_cpp_engine": pricer_bindings.using_cpp_engine(),
        "total_elapsed_sec": total_elapsed,
        "per_symbol": per_symbol,
    }


def _run_as_subprocess(backend: str, db_path: str, symbols: list[str]) -> dict:
    env_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        backend,
        "--db",
        db_path,
        "--symbols",
        ",".join(symbols),
    ]
    result = subprocess.run(env_args, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"{backend} worker subprocess failed:\n{result.stderr}")
    # The worker prints exactly one JSON line as its last line of stdout;
    # everything before that (if any) is incidental (e.g. library warnings).
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/db/quant.duckdb")
    parser.add_argument("--symbols", default="NIFTY,BANKNIFTY")
    parser.add_argument("--_worker", choices=["cpp", "fallback"], default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args._worker == "fallback":
        sys.modules["core.qengine"] = None  # force ImportError in pricer_bindings/svi_surface
        result = _run_worker(args.db, symbols)
        print(json.dumps(result))
        return
    if args._worker == "cpp":
        result = _run_worker(args.db, symbols)
        print(json.dumps(result))
        return

    # Top-level invocation: run both backends as subprocesses and compare.
    print(f"Running C++ backend against {args.db} ({', '.join(symbols)})...")
    cpp_result = _run_as_subprocess("cpp", args.db, symbols)
    if not cpp_result["using_cpp_engine"]:
        print(
            "WARNING: core/qengine*.so isn't built in this environment -- the 'C++' run actually "
            "used the fallback too, so this comparison is not meaningful. Build it first: "
            "cmake -S csrc -B csrc/build && cmake --build csrc/build -j$(nproc)"
        )

    print(f"Running Python fallback backend (Numba + SciPy) against {args.db} ({', '.join(symbols)})...")
    fallback_result = _run_as_subprocess("fallback", args.db, symbols)
    if fallback_result["using_cpp_engine"]:
        raise RuntimeError("fallback worker unexpectedly used the C++ engine -- poisoning did not take effect")

    print()
    print(f"{'Symbol':<12}{'C++ (s)':>12}{'Fallback (s)':>16}{'Speedup':>10}")
    for symbol in symbols:
        c = cpp_result["per_symbol"][symbol]
        f = fallback_result["per_symbol"][symbol]
        speedup = f["elapsed_sec"] / c["elapsed_sec"] if c["elapsed_sec"] > 0 else float("nan")
        print(f"{symbol:<12}{c['elapsed_sec']:>12.2f}{f['elapsed_sec']:>16.2f}{speedup:>9.2f}x")
    total_speedup = fallback_result["total_elapsed_sec"] / cpp_result["total_elapsed_sec"]
    print(f"{'TOTAL':<12}{cpp_result['total_elapsed_sec']:>12.2f}{fallback_result['total_elapsed_sec']:>16.2f}{total_speedup:>9.2f}x")

    print()
    print("Correctness sanity check (fitted expiries / converged / mean RMSE per symbol):")
    for symbol in symbols:
        c = cpp_result["per_symbol"][symbol]
        f = fallback_result["per_symbol"][symbol]
        print(
            f"  {symbol}: C++ {c['fitted_expiries']} fitted / {c['converged']} converged / "
            f"errors={c['errors']} / mean_rmse={c['mean_rmse']:.4e}"
        )
        print(
            f"  {symbol}: fallback {f['fitted_expiries']} fitted / {f['converged']} converged / "
            f"errors={f['errors']} / mean_rmse={f['mean_rmse']:.4e}"
        )


if __name__ == "__main__":
    main()
