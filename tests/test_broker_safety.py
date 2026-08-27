"""Regression guards for this project's hard safety rules around live
broker integrations (see CLAUDE.md and docs/WHITEPAPER.md "Safety & scope
boundaries"):

  1. This project is strictly read-only against broker APIs -- market data
     login and touchline/depth subscription only. It must never place,
     modify, or cancel an order. `MarketDataInterface` intentionally
     exposes only data methods; these tests fail loudly if that contract
     -- or a concrete adapter -- ever grows order-placement capability.
  2. Broker credentials are never committed to the repo and never logged.

These are deliberately blunt (an abstract-method-set check, a keyword
scan) rather than a full static-analysis pass -- the goal is to make an
accidental regression (someone, human or AI, adding a `place_order()` call
while wiring up a future feature) impossible to land silently, not to
catch every conceivable circumvention.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from core.market_data_interface import MarketDataInterface

REPO_ROOT = Path(__file__).resolve().parents[1]

_BROKER_ADAPTER_FILES = [
    REPO_ROOT / "core" / "shoonya_ws_adapter.py",
    REPO_ROOT / "core" / "upstox_protobuf_adapter.py",
    REPO_ROOT / "core" / "nse_python_adapter.py",
    REPO_ROOT / "main.py",
]

# Case-insensitive substrings that would indicate order-placement,
# modification, or cancellation capability -- NorenApi and Upstox endpoint/
# method names, plus generic verbs no market-data adapter has a legitimate
# reason to contain.
_FORBIDDEN_SUBSTRINGS = [
    "placeorder",
    "modifyorder",
    "cancelorder",
    "exitorder",
    "place_order",
    "modify_order",
    "cancel_order",
    "exit_order",
    "punch_order",
    "/order/place",
    "/v2/order",
    "v3/order/place",
    "norenorder",
    "def buy(",
    "def sell(",
    "transactiontype",
]


def test_market_data_interface_exposes_only_data_methods():
    assert set(MarketDataInterface.__abstractmethods__) == {
        "connect",
        "disconnect",
        "get_option_chain",
        "subscribe",
        "is_live",
    }


def test_broker_adapters_and_entrypoint_contain_no_order_placement_code():
    hits: list[str] = []
    for path in _BROKER_ADAPTER_FILES:
        text = path.read_text().lower()
        for needle in _FORBIDDEN_SUBSTRINGS:
            if needle in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: found forbidden substring {needle!r}")
    assert not hits, "Order-placement code detected where only market-data code is allowed:\n" + "\n".join(hits)


def test_shoonya_credentials_have_no_hardcoded_defaults():
    # Every secret field must be a required constructor argument -- no
    # default value that could silently mask a missing/placeholder credential.
    from core.shoonya_ws_adapter import ShoonyaCredentials

    secret_fields = {"user_id", "password", "totp_secret", "vendor_code", "api_key", "imei"}
    defaults = {f.name: f.default for f in ShoonyaCredentials.__dataclass_fields__.values()}
    for field_name in secret_fields:
        assert defaults[field_name] is dataclasses.MISSING, f"ShoonyaCredentials.{field_name} must not have a default value"


def test_env_files_are_gitignored_but_the_example_template_is_not():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert ".env" in gitignore.splitlines()
    assert ".env.*" in gitignore.splitlines()
    assert "!.env.example" in gitignore.splitlines()
    assert (REPO_ROOT / ".env.example").exists()


def test_env_example_template_has_no_real_looking_values():
    example = (REPO_ROOT / ".env.example").read_text()
    for line in example.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        _, _, value = line.partition("=")
        assert value.strip() == "", f"'.env.example' must ship with empty placeholder values, found: {line!r}"
