#!/usr/bin/env bash
# Thin entrypoint: ensures the DuckDB volume mount point exists (first run
# on a fresh volume) before handing off to main.py's CLI.
set -euo pipefail

mkdir -p /app/data/db

exec python main.py "$@"
