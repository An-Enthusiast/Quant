# Quant

Real-Time Options Market Making & Non-Linear Risk Hedging Engine for NSE
index options (Nifty, BankNifty). A low-latency C++ Black-Scholes/SVI
pricing core, Avellaneda-Stoikov quoting with an ML toxicity overlay,
multi-asset risk aggregation with automated delta hedging, and an
event-driven backtester, behind a `MarketDataInterface` abstraction that
takes the data source from a zero-cost polling prototype to a live broker
feed without touching any pricer/alpha/risk code.

See **[docs/WHITEPAPER.md](docs/WHITEPAPER.md)** for the full architecture,
mathematical derivations, measured throughput, backtest results, and a
step-by-step setup guide.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cmake -S csrc -B csrc/build -DCMAKE_BUILD_TYPE=Release
cmake --build csrc/build -j"$(nproc)"

pytest -q

python main.py --mode prototype --fixture --symbols NIFTY --max-polls 1
python -m backtest.run_backtest --source synthetic --n-ticks 180
```

## Layout

| Path | Contents |
|---|---|
| `csrc/` | C++ Newton-Raphson Black-Scholes solver + SVI calibration (pybind11) |
| `core/` | Data model, market-data adapters, pricer bindings, vol surface |
| `data/` | DuckDB schema + ingestion |
| `alpha/` | Avellaneda-Stoikov quoting, order-flow toxicity classifier |
| `risk/` | Portfolio Greeks aggregation, risk limits, automated hedging |
| `backtest/` | Event-driven tick-replay simulator |
| `deployment/` | Multi-stage Dockerfile + docker-compose |
| `docs/` | Technical whitepaper |
| `tests/` | pytest suite |
