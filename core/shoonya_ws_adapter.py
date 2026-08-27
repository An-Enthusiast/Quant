"""Phase 2: Shoonya (Finvasia) NorenApi websocket adapter for live L2 tick
streaming.

Implements the same `MarketDataInterface` contract as `NSEPythonAdapter`
(core/nse_python_adapter.py), so switching `main.py --mode live --broker
shoonya` in requires zero changes to core/pricer_bindings.py,
alpha/quote_engine.py, risk/*, or backtest/*.

**Status.** Built against Shoonya's publicly documented NorenApi REST/websocket
protocol (the same backend a number of Indian brokers -- Finvasia/Shoonya,
Flattrade, and others -- run under different branding), without a live
account to test against yet (see docs/WHITEPAPER.md §9). Everything here is
therefore verified by unit test against recorded/synthetic payloads shaped
like the documented schema, not against a real login/tick. Two pieces are
flagged below as needing reconfirmation against a live account before this
adapter is trusted in production:

  1. `_SYMBOL_MASTER_URL`'s CSV column names (`_parse_symbol_master_csv`).
  2. `_INDEX_TOKENS`, the well-known NSE index tokens used to source `spot`.

Protocol summary (Noren WS Trading Protocol, publicly documented):
  1. REST login (`QuickAuth`): POST a `jData` JSON payload (sha256-hashed
     password, a fresh TOTP code, sha256(uid|api_key) as `appkey`) and get
     back a `susertoken` session token.
  2. Open the websocket at `credentials.ws_endpoint` and send a `{"t":"c",
     ...}` connect frame carrying that token; the server acks with
     `{"t":"ck","s":"OK"}`.
  3. Subscribe to touchline updates for a set of `EXCHANGE|token` instrument
     keys via `{"t":"t","k":"NFO|26000#NFO|26009"}`.
  4. Receive `{"t":"tk",...}` (full touchline snapshot) then `{"t":"tf",...}`
     (incremental fields-only updates) per subscribed token, merged onto the
     last known `OptionContract` for that (expiry, strike, option_type).

Resolving which NFO instrument tokens correspond to a given underlying's
option chain requires Shoonya's daily symbol master file (`_SYMBOL_MASTER_URL`);
resolving the underlying's own spot price requires subscribing to its NSE
index token separately (options live on the NFO segment, the index itself on
NSE) -- both handled in `subscribe()` / `_ensure_symbol_master()`.

Ticks arrive on a background thread (the `websocket-client` run_forever loop
started in `connect()`), so registered `subscribe()` callbacks -- and any
code reading `get_option_chain()` concurrently -- must be safe to run off
the main thread; `_lock` guards the shared per-symbol contract state.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pyotp
import requests

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover - exercised via _open_websocket's own check
    websocket = None

from core.market_data_interface import MarketDataInterface
from core.option_chain import OptionChainSnapshot, OptionContract, OptionType

logger = logging.getLogger(__name__)

_LOGIN_URL = "https://api.shoonya.com/NorenWClientTP/QuickAuth"
_SYMBOL_MASTER_URL = "https://api.shoonya.com/NFO_symbols.txt.zip"
_NFO_EXCHANGE = "NFO"
_INDEX_EXCHANGE = "NSE"
_OPTION_INSTRUMENT_TYPES = {"OPTIDX"}

# Well-known Noren-backend NSE index tokens (stable across the NorenApi
# brokers that publish them; unverified against a live Shoonya symbol
# master in this environment -- see module docstring).
_INDEX_TOKENS = {"NIFTY": "26000", "BANKNIFTY": "26009"}


@dataclass(slots=True)
class ShoonyaCredentials:
    user_id: str
    password: str
    totp_secret: str
    vendor_code: str
    api_key: str
    imei: str
    ws_endpoint: str = "wss://api.shoonya.com/NorenWSTP/"


@dataclass(slots=True)
class _InstrumentMeta:
    token: str
    symbol: str
    expiry: date
    strike: float
    option_type: OptionType


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_login_payload(credentials: ShoonyaCredentials, *, totp_code: str | None = None) -> dict:
    """Build the QuickAuth `jData` payload.

    `totp_code` is injectable so tests can supply a fixed code; production
    calls always derive it live from `credentials.totp_secret` (a TOTP code
    is single-use and expires in ~30s, so it cannot be precomputed).
    """
    code = totp_code if totp_code is not None else pyotp.TOTP(credentials.totp_secret).now()
    return {
        "apkversion": "1.0.0",
        "uid": credentials.user_id,
        "pwd": _sha256(credentials.password),
        "factor2": code,
        "vc": credentials.vendor_code,
        "appkey": _sha256(f"{credentials.user_id}|{credentials.api_key}"),
        "imei": credentials.imei,
        "source": "API",
    }


def _parse_expiry(raw: str) -> date:
    """Shoonya's symbol master reports expiry as `DD-MMM-YYYY` (e.g.
    `28-MAR-2024`); `%b` matches month abbreviations case-insensitively.
    """
    return datetime.strptime(raw.strip(), "%d-%b-%Y").date()


def parse_symbol_master_csv(csv_text: str, symbols: set[str]) -> list[_InstrumentMeta]:
    """Parse Shoonya's NFO symbol-master CSV into option instrument metadata,
    filtered to `symbols` (e.g. `{"NIFTY", "BANKNIFTY"}`) and index options
    only (`Instrument == "OPTIDX"` -- excludes futures and stock options).

    Rows with an unparsable expiry/strike/option-type are skipped with a
    warning rather than failing the whole load -- one malformed row (a stale
    or partially-updated master file) shouldn't block every other
    instrument from resolving.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    out: list[_InstrumentMeta] = []
    for row in reader:
        symbol = (row.get("Symbol") or "").strip().upper()
        if symbol not in symbols:
            continue
        if (row.get("Instrument") or "").strip().upper() not in _OPTION_INSTRUMENT_TYPES:
            continue
        try:
            out.append(
                _InstrumentMeta(
                    token=(row["Token"]).strip(),
                    symbol=symbol,
                    expiry=_parse_expiry(row["Expiry"]),
                    strike=float(row["StrikePrice"]),
                    option_type=OptionType((row.get("OptionType") or "").strip().upper()),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("skipping unparsable Shoonya symbol-master row %r: %s", row, exc)
    return out


def _maybe_float(tick: dict, key: str) -> float | None:
    raw = tick.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _maybe_int(tick: dict, key: str) -> int | None:
    val = _maybe_float(tick, key)
    return None if val is None else int(val)


class ShoonyaWebSocketAdapter(MarketDataInterface):
    """Live L2 tick adapter over Shoonya's NorenApi websocket feed. Same
    `MarketDataInterface` contract as `NSEPythonAdapter` -- swapping this in
    for Phase 2 requires zero changes to `core/pricer_bindings.py`,
    `alpha/quote_engine.py`, `risk/*`, or `backtest/*`.
    """

    def __init__(
        self,
        credentials: ShoonyaCredentials,
        max_reconnect_backoff_sec: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.credentials = credentials
        self.max_reconnect_backoff_sec = max_reconnect_backoff_sec
        self._session = session or requests.Session()
        self._session_token: str | None = None
        self._ws: Any = None
        self._ws_thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._lock = threading.RLock()
        self._instruments_by_token: dict[str, _InstrumentMeta] = {}
        self._contracts: dict[str, dict[tuple[date, float, OptionType], OptionContract]] = {}
        self._prev_day_oi: dict[str, dict[tuple[date, float, OptionType], int]] = {}
        self._spot: dict[str, float] = {}
        self._latest_snapshots: dict[str, OptionChainSnapshot] = {}
        self._callbacks: list[Callable[[OptionChainSnapshot], None]] = []

    @property
    def is_live(self) -> bool:
        return True

    def connect(self) -> None:
        payload = build_login_payload(self.credentials)
        resp = self._session.post(_LOGIN_URL, data="jData=" + json.dumps(payload), timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if body.get("stat") != "Ok":
            # Only ever surface the server's own "emsg" field, never the raw
            # response body -- an unexpected/malformed response could in
            # principle echo back request data, and this message may end up
            # in logs or an exception report.
            raise RuntimeError(f"Shoonya login failed: {body.get('emsg', 'no error message provided by server')}")
        self._session_token = body["susertoken"]
        self._open_websocket()

    def _open_websocket(self) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is not installed; run `pip install -r requirements.txt`."
            )
        self._connected.clear()
        self._ws = websocket.WebSocketApp(
            self.credentials.ws_endpoint,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        if not self._connected.wait(timeout=10):
            raise RuntimeError("Shoonya websocket did not acknowledge the connect frame within 10s")

    def _on_ws_open(self, ws: Any) -> None:
        ws.send(
            json.dumps(
                {
                    "t": "c",
                    "uid": self.credentials.user_id,
                    "actid": self.credentials.user_id,
                    "source": "API",
                    "susertoken": self._session_token,
                }
            )
        )

    def _on_ws_message(self, ws: Any, message: str) -> None:
        self._dispatch_message(json.loads(message))

    def _on_ws_error(self, ws: Any, error: Exception) -> None:
        logger.error("Shoonya websocket error: %s", error)

    def _on_ws_close(self, ws: Any, status_code: int | None, msg: str | None) -> None:
        self._connected.clear()
        logger.warning("Shoonya websocket closed: %s %s", status_code, msg)

    def disconnect(self) -> None:
        if self._ws is not None:
            self._ws.close()
        self._session_token = None
        self._connected.clear()

    def get_option_chain(self, symbol: str) -> OptionChainSnapshot:
        try:
            return self._latest_snapshots[symbol]
        except KeyError as exc:
            raise RuntimeError(
                f"No snapshot cached yet for {symbol}; subscribe() must receive at least one "
                "tick before get_option_chain() can return data."
            ) from exc

    def subscribe(self, symbols: list[str], callback: Callable[[OptionChainSnapshot], None]) -> None:
        if self._ws is None or not self._connected.is_set():
            raise RuntimeError("subscribe() requires connect() to have established a live websocket session first.")
        self._callbacks.append(callback)
        for symbol in symbols:
            self._ensure_symbol_master(symbol)
            tokens = [meta.token for meta in self._instruments_by_token.values() if meta.symbol == symbol]
            if not tokens:
                raise RuntimeError(f"No NFO option instrument tokens resolved for {symbol}.")
            index_token = _INDEX_TOKENS.get(symbol)
            keys = [f"{_NFO_EXCHANGE}|{tok}" for tok in tokens]
            if index_token is not None:
                keys.append(f"{_INDEX_EXCHANGE}|{index_token}")
            self._contracts.setdefault(symbol, {})
            self._ws.send(json.dumps({"t": "t", "k": "#".join(keys)}))

    def _ensure_symbol_master(self, symbol: str) -> None:
        if any(meta.symbol == symbol for meta in self._instruments_by_token.values()):
            return
        resp = self._session.get(_SYMBOL_MASTER_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
        for meta in parse_symbol_master_csv(csv_text, {symbol}):
            self._instruments_by_token[meta.token] = meta

    def _dispatch_message(self, msg: dict) -> None:
        msg_type = msg.get("t")
        if msg_type == "ck":
            if msg.get("s") == "OK":
                self._connected.set()
            else:
                logger.error("Shoonya websocket connect ack failed: %s", msg)
        elif msg_type in ("tk", "tf"):
            self._on_tick(msg)

    def _on_tick(self, tick: dict) -> None:
        token = tick.get("tk")
        if token is None:
            return

        for symbol, index_token in _INDEX_TOKENS.items():
            if token == index_token and tick.get("e") == _INDEX_EXCHANGE:
                lp = _maybe_float(tick, "lp")
                if lp is not None:
                    with self._lock:
                        self._spot[symbol] = lp
                    self._publish_snapshot(symbol)
                return

        meta = self._instruments_by_token.get(token)
        if meta is None:
            return  # tick for a token we haven't resolved via the symbol master

        key = (meta.expiry, meta.strike, meta.option_type)
        with self._lock:
            contracts = self._contracts.setdefault(meta.symbol, {})
            prev_day_oi_map = self._prev_day_oi.setdefault(meta.symbol, {})
            # "poi" (previous trading day's closing OI) typically arrives once,
            # on the initial "tk" packet; cache it so change_in_oi (= oi - poi,
            # matching NSE's own changeinOpenInterest definition, core/option_chain.py)
            # stays correct across later "tf" partial updates that omit it.
            new_poi = _maybe_int(tick, "poi")
            if new_poi is not None:
                prev_day_oi_map[key] = new_poi
            known_poi = prev_day_oi_map.get(key)

            existing = contracts.get(key)
            ts = datetime.fromtimestamp(int(tick["ft"])) if tick.get("ft") else datetime.now()
            if existing is None:
                oi_val = _maybe_int(tick, "oi") or 0
                existing = OptionContract(
                    symbol=meta.symbol,
                    expiry=meta.expiry,
                    strike=meta.strike,
                    option_type=meta.option_type,
                    ltp=_maybe_float(tick, "lp") or 0.0,
                    bid=_maybe_float(tick, "bp1") or 0.0,
                    bid_qty=_maybe_int(tick, "bq1") or 0,
                    ask=_maybe_float(tick, "sp1") or 0.0,
                    ask_qty=_maybe_int(tick, "sq1") or 0,
                    oi=oi_val,
                    change_in_oi=(oi_val - known_poi) if known_poi is not None else 0,
                    volume=_maybe_int(tick, "v") or 0,
                    timestamp=ts,
                )
                contracts[key] = existing
            else:
                for field_name, tick_key, caster in (
                    ("ltp", "lp", _maybe_float),
                    ("bid", "bp1", _maybe_float),
                    ("bid_qty", "bq1", _maybe_int),
                    ("ask", "sp1", _maybe_float),
                    ("ask_qty", "sq1", _maybe_int),
                    ("oi", "oi", _maybe_int),
                    ("volume", "v", _maybe_int),
                ):
                    value = caster(tick, tick_key)
                    if value is not None:
                        setattr(existing, field_name, value)
                if known_poi is not None:
                    existing.change_in_oi = existing.oi - known_poi
                existing.timestamp = ts

        self._publish_snapshot(meta.symbol)

    def _publish_snapshot(self, symbol: str) -> None:
        with self._lock:
            contracts = list(self._contracts.get(symbol, {}).values())
            spot = self._spot.get(symbol, 0.0)
        if not contracts:
            return
        snapshot = OptionChainSnapshot(
            symbol=symbol, timestamp=datetime.now(), spot=spot, contracts=contracts
        )
        self._latest_snapshots[symbol] = snapshot
        for callback in self._callbacks:
            callback(snapshot)

    def _run_forever(self) -> None:
        backoff = 1.0
        while True:
            try:
                self.connect()
                backoff = 1.0
                return
            except Exception:
                logger.exception("ShoonyaWebSocketAdapter: connection failed, retrying in %.1fs", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_backoff_sec)
