"""Safety-gate tests for scripts/live_pull_smoke_test.py.

These deliberately do NOT exercise the live network path (that's the one
thing this script is for, and it isn't something CI/tests should trigger
implicitly). What's tested here is the safety contract: the script refuses
to run without an explicit acknowledgement flag, and it never writes to a
path anything else in the repo depends on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "live_pull_smoke_test.py"


def test_refuses_to_run_without_explicit_flag():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--symbol", "NIFTY"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=10,
    )
    assert result.returncode != 0
    assert "--yes-hit-live-nse" in result.stderr


def test_safe_db_path_is_isolated_from_other_data_paths():
    import scripts.live_pull_smoke_test as harness

    assert harness.SAFE_DB_PATH == "data/db/live_smoke_test.duckdb"
    # Must not collide with the default dev DB or anything under sample_data/
    # that tests/ and the fixture-mode adapter rely on.
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
    assert "--yes-hit-live-nse" in result.stdout
