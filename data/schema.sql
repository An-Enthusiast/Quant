-- DuckDB schema for L2-ish option-chain snapshots.
--
-- One row per (timestamp, symbol, expiry, strike, option_type) quote.
-- Append-only: ingestion always INSERTs new snapshot rows, never UPDATEs,
-- so history is a full time series usable for both the backtester and
-- future model retraining, and DuckDB's columnar storage keeps large
-- histories out-of-core rather than requiring everything to fit in memory.

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    ts                  TIMESTAMP     NOT NULL,  -- exchange-reported snapshot timestamp
    ingested_at         TIMESTAMP     NOT NULL,  -- wall-clock time this row was written
    symbol              VARCHAR       NOT NULL,  -- underlying, e.g. 'NIFTY' / 'BANKNIFTY'
    expiry              DATE          NOT NULL,
    strike              DOUBLE        NOT NULL,
    option_type         VARCHAR(2)    NOT NULL,  -- 'CE' / 'PE'
    underlying_spot      DOUBLE        NOT NULL,
    ltp                 DOUBLE        NOT NULL,
    bid                 DOUBLE        NOT NULL,
    bid_qty             BIGINT        NOT NULL,
    ask                 DOUBLE        NOT NULL,
    ask_qty             BIGINT        NOT NULL,
    oi                  BIGINT        NOT NULL,
    change_in_oi        BIGINT        NOT NULL,
    volume              BIGINT        NOT NULL,
    exchange_iv         DOUBLE                    -- exchange-reported IV, if any (nullable)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_ts ON option_chain_snapshots (symbol, ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_expiry ON option_chain_snapshots (symbol, expiry);
