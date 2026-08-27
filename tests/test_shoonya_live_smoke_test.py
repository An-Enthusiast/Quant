"""Safety-gate tests for scripts/shoonya_live_smoke_test.py.

Mirrors tests/test_live_pull_smoke_test.py's approach: never exercise the
live network/websocket path here, only the safety contract -- the script
refuses to run without an explicit flag or with missing credentials, and it
never writes to a path anything else in the repo depends on.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "shoonya_live_smoke_test.py"


def test_refuses_to_run_without_explicit_flag():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--symbol", "NIFTY"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=10,
    )
    assert result.returncode != 0
    assert "--yes-hit-live-shoonya" in result.stderr


def test_refuses_to_run_without_credentials_and_never_prints_secret_values():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHOONYA_")}
    env["SHOONYA_USER_ID"] = "present-but-incomplete"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--yes-hit-live-shoonya", "--symbol", "NIFTY"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=10,
        env=env,
    )
    assert result.returncode != 0
    assert "SHOONYA_PASSWORD" in result.stderr
    assert "SHOONYA_TOTP_SECRET" in result.stderr
    assert "present-but-incomplete" not in result.stderr  # never echo credential values


def test_safe_db_path_is_isolated_from_other_data_paths():
    import scripts.shoonya_live_smoke_test as harness

    assert harness.SAFE_DB_PATH == "data/db/shoonya_live_smoke_test.duckdb"
    assert harness.SAFE_DB_PATH != "data/db/quant.duckdb"
    assert "sample_data" not in harness.SAFE_DB_PATH


def test_help_does_not_require_network_and_lists_safety_flag():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=10,
    )
    assert result.returncode == 0
    assert "--yes-hit-live-shoonya" in result.stdout


def test_credentials_from_env_reads_all_fields(monkeypatch):
    import scripts.shoonya_live_smoke_test as harness

    monkeypatch.setenv("SHOONYA_USER_ID", "FA123")
    monkeypatch.setenv("SHOONYA_PASSWORD", "pw")
    monkeypatch.setenv("SHOONYA_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("SHOONYA_VENDOR_CODE", "FA123_U")
    monkeypatch.setenv("SHOONYA_API_KEY", "key123")
    monkeypatch.delenv("SHOONYA_IMEI", raising=False)

    creds = harness.credentials_from_env()
    assert creds.user_id == "FA123"
    assert creds.vendor_code == "FA123_U"
    assert creds.imei  # defaulted, non-empty
