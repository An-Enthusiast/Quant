# Instructions for Claude Code working in this repository

Two hard rules for any work touching broker integrations (Shoonya,
Upstox, or any future broker adapter). Both are enforced by
`tests/test_broker_safety.py` -- if a change trips that suite, the change
violates one of these rules and must be fixed, not the test.

## 1. Broker credentials are never committed, logged, or printed

- Credentials (user id, password, TOTP secret, API/app key, access
  tokens, session tokens) live **only** in environment variables, read at
  call time. Never hardcode a real value anywhere in the repo, including
  in commit messages, test fixtures, or scratch files that might get
  committed.
- `.env` and `.env.*` are gitignored (`.gitignore`); `.env.example` is the
  only tracked template and must always ship with empty placeholder
  values (`KEY=`, never `KEY=realvalue`).
- Never log, print, or include in an exception message a raw credential,
  session/access token, or an unfiltered server response body that might
  contain one. Error messages should name *what* failed (e.g. which env
  var is missing, the server's own short error string) and nothing more.
- Before committing, check `git status`/`git diff` for anything that
  looks like a real secret -- especially in files that don't obviously
  look credential-related.

## 2. Market data only -- never place, modify, or cancel a live order

This project reads option-chain/quote data from broker APIs. It does
**not**, and must never, execute trades. Concretely:

- Do not implement, call, or wire up any order-placement, order-
  modification, order-cancellation, or position-exit endpoint on any
  broker API (Shoonya/NorenApi's `PlaceOrder`/`ModifyOrder`/
  `CancelOrder`, Upstox's `/v2/order` or `/v3/order/place`, or any
  equivalent on a future broker).
- `core/market_data_interface.py`'s `MarketDataInterface` ABC is
  intentionally scoped to `connect` / `disconnect` / `get_option_chain` /
  `subscribe` / `is_live` only. Do not add trading methods to it, and do
  not add a parallel "order" interface without the repository owner
  explicitly asking for one, in writing, in the task itself.
- `risk/hedging_engine.py`'s `compute_hedge_orders` and the backtester's
  `backtest/execution_sim.py` compute/simulate hedge sizing and fills --
  they do not, and must not, call a live broker to execute anything. Keep
  it that way: these stay analytics/simulation, not execution.
- If a task genuinely requires order execution, stop and ask the user
  first -- do not infer that a market-data or hedging task implies
  authorization to place real trades.

See `docs/WHITEPAPER.md`'s "Safety & scope boundaries" section for the
fuller rationale and current implementation status.
