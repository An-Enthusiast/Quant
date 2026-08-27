"""Tests for core/shoonya_ws_adapter.py.

No real Shoonya account/network access is available (docs/WHITEPAPER.md §9),
so these exercise the login-payload construction, symbol-master parsing, and
tick-to-OptionContract mapping against fixture-shaped payloads that follow
the documented NorenApi schema, plus the REST/websocket calls against small
fake stand-ins (constructor-injected `session=`, monkeypatched `websocket`
module) rather than a real network/socket -- consistent with how the rest
of this suite avoids live calls (fixture-mode adapters, recorded archives).
"""

from __future__ import annotations

import datetime
import json

import pytest

from core.option_chain import OptionType
from core.shoonya_ws_adapter import (
    ShoonyaCredentials,
    ShoonyaWebSocketAdapter,
    _InstrumentMeta,
    build_login_payload,
    parse_symbol_master_csv,
)

_CREDENTIALS = ShoonyaCredentials(
    user_id="FA123",
    password="secret",
    totp_secret="JBSWY3DPEHPK3PXP",
    vendor_code="FA123_U",
    api_key="apikey123",
    imei="abc123",
)

_SYMBOL_MASTER_CSV = """Exchange,Token,LotSize,Symbol,TradingSymbol,Expiry,Instrument,OptionType,StrikePrice,TickSize
NFO,35001,50,NIFTY,NIFTY28MAR24C22000,28-MAR-2024,OPTIDX,CE,22000,0.05
NFO,35002,50,NIFTY,NIFTY28MAR24P22000,28-MAR-2024,OPTIDX,PE,22000,0.05
NFO,35100,25,BANKNIFTY,BANKNIFTY28MAR24C47000,28-MAR-2024,OPTIDX,CE,47000,0.05
NFO,40000,50,NIFTY,NIFTY28MAR24FUT,28-MAR-2024,FUTIDX,,0,0.05
"""


class _FakeResponse:
    def __init__(self, *, json_body=None, content=b"", status_code=200):
        self._json_body = json_body
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Stands in for requests.Session; records calls, returns canned responses."""

    def __init__(self, login_response, get_response=None):
        self._login_response = login_response
        self._get_response = get_response
        self.posts: list[tuple[str, str]] = []
        self.gets: list[str] = []

    def post(self, url, data, timeout):
        self.posts.append((url, data))
        return self._login_response

    def get(self, url, timeout):
        self.gets.append(url)
        return self._get_response


class _FakeWebSocketApp:
    """Stands in for websocket.WebSocketApp: run_forever() synchronously
    fires on_open then a canned "ck" ack, so connect() completes without a
    real socket or background thread race.
    """

    last_instance: "_FakeWebSocketApp | None" = None

    def __init__(self, url, on_open=None, on_message=None, on_error=None, on_close=None):
        self.url = url
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.sent: list[str] = []
        _FakeWebSocketApp.last_instance = self

    def send(self, data):
        self.sent.append(data)

    def run_forever(self):
        if self.on_open:
            self.on_open(self)
        if self.on_message:
            self.on_message(self, json.dumps({"t": "ck", "s": "OK"}))

    def close(self):
        if self.on_close:
            self.on_close(self, 1000, "closed")


def _connected_adapter(monkeypatch) -> ShoonyaWebSocketAdapter:
    import core.shoonya_ws_adapter as mod

    monkeypatch.setattr(mod, "websocket", type("_W", (), {"WebSocketApp": _FakeWebSocketApp}))
    session = _FakeSession(_FakeResponse(json_body={"stat": "Ok", "susertoken": "tok123"}))
    adapter = ShoonyaWebSocketAdapter(_CREDENTIALS, session=session)
    adapter.connect()
    return adapter


def test_build_login_payload_hashes_password_and_appkey_and_includes_totp():
    payload = build_login_payload(_CREDENTIALS, totp_code="654321")
    assert payload["uid"] == "FA123"
    assert payload["vc"] == "FA123_U"
    assert payload["factor2"] == "654321"
    assert payload["pwd"] != "secret"  # sha256-hashed, not plaintext
    assert len(payload["pwd"]) == 64
    assert len(payload["appkey"]) == 64


def test_parse_symbol_master_csv_filters_by_symbol_and_instrument_type():
    metas = parse_symbol_master_csv(_SYMBOL_MASTER_CSV, {"NIFTY"})
    assert len(metas) == 2  # the BANKNIFTY row and the FUTIDX row are excluded
    assert {m.option_type for m in metas} == {OptionType.CALL, OptionType.PUT}
    assert all(m.symbol == "NIFTY" and m.strike == 22000.0 for m in metas)
    assert metas[0].expiry == datetime.date(2024, 3, 28)


def test_parse_symbol_master_csv_skips_unparsable_rows_without_raising():
    bad_csv = _SYMBOL_MASTER_CSV + "NFO,35999,50,NIFTY,BROKEN,not-a-date,OPTIDX,CE,22500,0.05\n"
    metas = parse_symbol_master_csv(bad_csv, {"NIFTY"})
    assert len(metas) == 2  # the broken row is skipped, not fatal


def test_connect_raises_clear_error_on_login_failure():
    session = _FakeSession(_FakeResponse(json_body={"stat": "Not_Ok", "emsg": "Invalid Password"}))
    adapter = ShoonyaWebSocketAdapter(_CREDENTIALS, session=session)
    with pytest.raises(RuntimeError, match="Invalid Password"):
        adapter.connect()


def test_connect_success_opens_websocket_and_sets_connected(monkeypatch):
    adapter = _connected_adapter(monkeypatch)
    assert adapter._connected.is_set()
    assert adapter._session_token == "tok123"
    sent = json.loads(_FakeWebSocketApp.last_instance.sent[0])
    assert sent == {"t": "c", "uid": "FA123", "actid": "FA123", "source": "API", "susertoken": "tok123"}


def test_subscribe_before_connect_raises():
    adapter = ShoonyaWebSocketAdapter(_CREDENTIALS)
    with pytest.raises(RuntimeError, match="connect"):
        adapter.subscribe(["NIFTY"], lambda snap: None)


def test_get_option_chain_before_any_tick_raises(monkeypatch):
    adapter = _connected_adapter(monkeypatch)
    with pytest.raises(RuntimeError, match="No snapshot cached"):
        adapter.get_option_chain("NIFTY")


def test_subscribe_resolves_symbol_master_and_sends_touchline_request(monkeypatch):
    adapter = _connected_adapter(monkeypatch)
    adapter._session._get_response = _FakeResponse(content=_zip_bytes(_SYMBOL_MASTER_CSV))

    received = []
    adapter.subscribe(["NIFTY"], lambda snap: received.append(snap))

    sub_msg = json.loads(_FakeWebSocketApp.last_instance.sent[-1])
    assert sub_msg["t"] == "t"
    keys = sub_msg["k"].split("#")
    assert "NFO|35001" in keys and "NFO|35002" in keys
    assert "NSE|26000" in keys  # NIFTY spot index token


def test_tick_mapping_updates_contract_and_computes_change_in_oi_from_poi(monkeypatch):
    adapter = _connected_adapter(monkeypatch)
    adapter._instruments_by_token["35001"] = _InstrumentMeta(
        token="35001", symbol="NIFTY", expiry=datetime.date(2024, 3, 28), strike=22000.0, option_type=OptionType.CALL
    )
    received = []
    adapter._callbacks.append(lambda snap: received.append(snap))

    adapter._dispatch_message({"t": "tk", "e": "NSE", "tk": "26000", "lp": "22150.5"})
    adapter._dispatch_message(
        {
            "t": "tk",
            "e": "NFO",
            "tk": "35001",
            "lp": "120.5",
            "bp1": "119.0",
            "sp1": "121.0",
            "bq1": "500",
            "sq1": "400",
            "oi": "1000000",
            "poi": "900000",
            "v": "50000",
        }
    )
    # partial update: only lp and oi change, everything else must be preserved
    adapter._dispatch_message({"t": "tf", "e": "NFO", "tk": "35001", "lp": "122.0", "oi": "1050000"})

    snap = adapter.get_option_chain("NIFTY")
    assert snap.spot == 22150.5
    assert len(snap.contracts) == 1
    c = snap.contracts[0]
    assert c.ltp == 122.0
    assert c.bid == 119.0 and c.ask == 121.0  # untouched by the partial update
    assert c.oi == 1050000
    assert c.change_in_oi == 1050000 - 900000
    # the spot-only tick publishes nothing (no contracts resolved yet); each
    # of the 2 option ticks publishes once
    assert len(received) == 2


def _zip_bytes(csv_text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("NFO_symbols.txt", csv_text)
    return buf.getvalue()
