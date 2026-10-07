"""
Binance Spot TESTNET adapter (testnet.binance.vision).

Real order lifecycle with worthless funds. Safety properties:

  • Idempotent submission — newClientOrderId is the deterministic journal id.
    After a timeout we QUERY by that id before ever re-sending, so a retry can
    never create a second order (Binance also rejects a duplicate id that is
    still live with -2010).
  • Rate limiting — token bucket over request weights, honouring 429/418
    Retry-After. Budget from REST_WEIGHT_LIMIT_PER_MIN (default 1100 of
    Binance's 1200/min, leaving headroom for the market-data paths).
  • Exchange filters — quantity is quantized down to LOT_SIZE.stepSize and
    checked against (MIN_)NOTIONAL before submission; violations REJECT
    locally instead of burning an API call.
  • Partial fills — the actual executedQty / weighted average price is
    returned, so the caller books what really happened, never the intent.
  • Secrets — key/secret/signature are never logged. This module is sync
    (requests) because _execute_trade already runs in a worker thread.

Base URL is hardcoded to the testnet; this adapter cannot reach mainnet.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from config import (
    BINANCE_TESTNET_API_KEY,
    BINANCE_TESTNET_API_SECRET,
    BINANCE_TESTNET_REST,
    ORDER_MAX_RETRIES,
    ORDER_POLL_TIMEOUT_SECS,
    REST_WEIGHT_LIMIT_PER_MIN,
)
from execution.base import ExecutionAdapter, Fill, OrderIntent

log = logging.getLogger(__name__)

_TERMINAL = {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH"}


class TokenBucketLimiter:
    """Minute-window weight budget. Blocks (sleeps) until weight is available."""

    def __init__(self, weight_per_min: int, clock=time.monotonic, sleeper=time.sleep):
        self.capacity = float(weight_per_min)
        self.tokens = float(weight_per_min)
        self.refill_rate = float(weight_per_min) / 60.0   # tokens per second
        self._clock = clock
        self._sleep = sleeper
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self, weight: int) -> None:
        while True:
            with self._lock:
                now = self._clock()
                self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.refill_rate)
                self._last = now
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                deficit = weight - self.tokens
                wait = deficit / self.refill_rate
            self._sleep(min(wait, 5.0))


class BinanceTestnetAdapter(ExecutionAdapter):
    name = "testnet"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers["X-MBX-APIKEY"] = BINANCE_TESTNET_API_KEY
        self._limiter = TokenBucketLimiter(REST_WEIGHT_LIMIT_PER_MIN)
        self._filters: dict[str, dict[str, float]] = {}
        self._filters_lock = threading.Lock()

    # ── HTTP plumbing ────────────────────────────────────────────────────────

    def _sign(self, params: dict[str, Any]) -> str:
        query = urlencode(params)
        sig = hmac.new(
            BINANCE_TESTNET_API_SECRET.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={sig}"

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                 *, signed: bool = False, weight: int = 1,
                 timeout: float = 10.0) -> tuple[int, Any]:
        """Returns (http_status, parsed_json_or_text). Raises only on transport error."""
        self._limiter.acquire(weight)
        params = dict(params or {})
        url = f"{BINANCE_TESTNET_REST}{path}"
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            body = self._sign(params)
            if method == "GET":
                url = f"{url}?{body}"
                resp = self._session.get(url, timeout=timeout)
            else:
                resp = self._session.request(
                    method, url, data=body, timeout=timeout,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        else:
            resp = self._session.request(method, url, params=params, timeout=timeout)

        if resp.status_code in (429, 418):
            retry_after = float(resp.headers.get("Retry-After", "5") or 5)
            log.warning("testnet rate-limited (%s) — backing off %.0fs",
                        resp.status_code, retry_after)
            time.sleep(min(retry_after, 60.0))
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, resp.text

    # ── Exchange filters ─────────────────────────────────────────────────────

    def _symbol_filters(self, symbol: str) -> Optional[dict[str, float]]:
        with self._filters_lock:
            if symbol in self._filters:
                return self._filters[symbol]
        status, data = self._request(
            "GET", "/exchangeInfo", {"symbol": symbol}, weight=20,
        )
        if status != 200 or not isinstance(data, dict) or not data.get("symbols"):
            log.warning("testnet exchangeInfo unavailable for %s (HTTP %s)", symbol, status)
            return None
        info = data["symbols"][0]
        out = {"stepSize": 0.0, "minNotional": 0.0, "minQty": 0.0}
        for f in info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                out["stepSize"] = float(f.get("stepSize", 0) or 0)
                out["minQty"] = float(f.get("minQty", 0) or 0)
            elif f.get("filterType") in ("NOTIONAL", "MIN_NOTIONAL"):
                out["minNotional"] = float(
                    f.get("minNotional", f.get("notional", 0)) or 0)
        with self._filters_lock:
            self._filters[symbol] = out
        return out

    @staticmethod
    def quantize_qty(qty: float, step: float) -> float:
        """Round DOWN to the lot step so we never oversubmit."""
        if step <= 0:
            return qty
        return int(qty / step + 1e-12) * step

    # ── Order lifecycle ──────────────────────────────────────────────────────

    @staticmethod
    def _binance_side(action: str) -> str:
        # buy/cover acquire base asset; sell/short dispose of it.
        return "BUY" if action in ("buy", "cover") else "SELL"

    def _fill_from_order_payload(self, payload: dict[str, Any]) -> Fill:
        status = str(payload.get("status", ""))
        executed = float(payload.get("executedQty", 0) or 0)
        cq = float(payload.get("cummulativeQuoteQty", 0) or 0)
        avg = (cq / executed) if executed > 0 else 0.0
        oid = str(payload.get("orderId", "")) or None
        if status == "FILLED" and executed > 0:
            return Fill("FILLED", executed, avg, oid)
        if executed > 0:
            return Fill("PARTIALLY_FILLED", executed, avg, oid, note=f"exchange status={status}")
        if status in _TERMINAL:
            return Fill("REJECTED", note=f"exchange status={status}")
        return Fill("FAILED", note=f"non-terminal status={status}")

    def _query_order(self, symbol: str, client_order_id: str) -> tuple[bool, Optional[dict]]:
        """(known, payload). known=False means the exchange has no such order."""
        status, data = self._request(
            "GET", "/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
            signed=True, weight=4,
        )
        if status == 200 and isinstance(data, dict):
            return True, data
        if isinstance(data, dict) and data.get("code") == -2013:   # does not exist
            return False, None
        return False, None if not isinstance(data, dict) else data

    def _poll_until_terminal(self, symbol: str, client_order_id: str) -> Optional[dict]:
        deadline = time.monotonic() + ORDER_POLL_TIMEOUT_SECS
        payload: Optional[dict] = None
        while time.monotonic() < deadline:
            known, payload = self._query_order(symbol, client_order_id)
            if known and payload and str(payload.get("status")) in _TERMINAL:
                return payload
            time.sleep(1.0)
        return payload

    def execute(self, intent: OrderIntent) -> Fill:
        filters = self._symbol_filters(intent.symbol)
        if filters is None:
            return Fill("REJECTED", note="symbol unavailable on testnet")

        qty = self.quantize_qty(float(intent.qty), filters["stepSize"])
        if qty <= 0 or qty < filters["minQty"]:
            return Fill("REJECTED", note=f"qty {intent.qty} below lot size")
        if filters["minNotional"] > 0 and qty * intent.ref_price < filters["minNotional"]:
            return Fill("REJECTED", note="below exchange min notional")

        params = {
            "symbol": intent.symbol,
            "side": self._binance_side(intent.action),
            "type": "MARKET",
            "quantity": f"{qty:.8f}".rstrip("0").rstrip("."),
            "newClientOrderId": intent.client_order_id,
            "newOrderRespType": "FULL",
        }

        attempts = 0
        while True:
            attempts += 1
            try:
                status, data = self._request("POST", "/order", params, signed=True, weight=1)
            except requests.RequestException as exc:
                log.warning("testnet order transport error for %s: %s — querying before retry",
                            intent.symbol, type(exc).__name__)
                known, payload = self._query_order(intent.symbol, intent.client_order_id)
                if known and payload:
                    # The order DID reach the exchange — adopt it, never resend.
                    terminal = (payload if str(payload.get("status")) in _TERMINAL
                                else self._poll_until_terminal(intent.symbol, intent.client_order_id))
                    return self._fill_from_order_payload(terminal or payload)
                if attempts > ORDER_MAX_RETRIES:
                    return Fill("FAILED", note=f"transport failed x{attempts}")
                time.sleep(min(2.0 ** attempts, 8.0))
                continue

            if status == 200 and isinstance(data, dict):
                if str(data.get("status")) in _TERMINAL:
                    return self._fill_from_order_payload(data)
                terminal = self._poll_until_terminal(intent.symbol, intent.client_order_id)
                return self._fill_from_order_payload(terminal or data)

            if isinstance(data, dict) and data.get("code") == -2010 \
                    and "duplicate" in str(data.get("msg", "")).lower():
                # Same clientOrderId already live — a prior attempt made it.
                known, payload = self._query_order(intent.symbol, intent.client_order_id)
                if known and payload:
                    terminal = (payload if str(payload.get("status")) in _TERMINAL
                                else self._poll_until_terminal(intent.symbol, intent.client_order_id))
                    return self._fill_from_order_payload(terminal or payload)
                return Fill("FAILED", note="duplicate id but order not found")

            note = data.get("msg") if isinstance(data, dict) else str(data)[:120]
            log.warning("testnet order rejected for %s: HTTP %s %s",
                        intent.symbol, status, note)
            return Fill("REJECTED", note=f"HTTP {status}: {note}")

    def reconcile(self, client_order_id: str) -> Optional[Fill]:
        # Journal rows carry the symbol; the caller passes it via closure —
        # kept simple: reconcile-by-id needs the symbol, so bot.py calls
        # reconcile_with_symbol instead for this adapter.
        return None

    def reconcile_with_symbol(self, symbol: str, client_order_id: str) -> Fill:
        known, payload = self._query_order(symbol, client_order_id)
        if not known or not payload:
            return Fill("REJECTED", note="not found on exchange (never executed)")
        return self._fill_from_order_payload(payload)
