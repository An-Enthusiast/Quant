"""Tests for the NSEPythonAdapter fixture-mode parsing (core/nse_python_adapter.py)."""

from __future__ import annotations

from core.nse_python_adapter import NSEPythonAdapter
from core.option_chain import OptionType


def test_fixture_mode_connects_and_parses_nifty():
    adapter = NSEPythonAdapter(use_fixture=True)
    adapter.connect()
    snapshot = adapter.get_option_chain("NIFTY")

    assert snapshot.symbol == "NIFTY"
    assert snapshot.spot > 0
    assert len(snapshot.contracts) > 0
    assert len(snapshot.expiries) >= 1

    for c in snapshot.contracts[:5]:
        assert c.strike > 0
        assert c.option_type in (OptionType.CALL, OptionType.PUT)


def test_fixture_mode_parses_banknifty():
    adapter = NSEPythonAdapter(use_fixture=True)
    adapter.connect()
    snapshot = adapter.get_option_chain("BANKNIFTY")
    assert snapshot.symbol == "BANKNIFTY"
    assert snapshot.spot > 40_000  # sanity: BankNifty trades well above Nifty


def test_get_option_chain_before_connect_raises():
    adapter = NSEPythonAdapter(use_fixture=True)
    try:
        adapter.get_option_chain("NIFTY")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_contract_mid_and_spread():
    adapter = NSEPythonAdapter(use_fixture=True)
    adapter.connect()
    snapshot = adapter.get_option_chain("NIFTY")
    c = next(c for c in snapshot.contracts if c.bid > 0 and c.ask > 0)
    assert c.mid == (c.bid + c.ask) / 2.0
    assert c.spread == c.ask - c.bid


def test_shoonya_and_upstox_adapters_raise_not_implemented_on_connect():
    from core.shoonya_ws_adapter import ShoonyaCredentials, ShoonyaWebSocketAdapter
    from core.upstox_protobuf_adapter import UpstoxCredentials, UpstoxProtobufAdapter

    shoonya = ShoonyaWebSocketAdapter(ShoonyaCredentials("u", "p", "t", "v", "k", "i"))
    assert shoonya.is_live
    try:
        shoonya.connect()
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass

    upstox = UpstoxProtobufAdapter(UpstoxCredentials("cid", "secret", "redirect"))
    assert upstox.is_live
    try:
        upstox.connect()
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
