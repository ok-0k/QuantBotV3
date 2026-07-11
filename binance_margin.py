"""
binance_margin.py — Live margin telemetry from Binance SAPI (V2).

When API keys are absent we synthesize utilization from the bot's internal cash/equity
state so risk limits still operate in paper / keyless mode.

Theory
------
Margin level ≈ assets / liabilities on borrowed funds. When level → 1, liquidation
imminent. We halt *new risk* when either live margin level is below a floor or
synthetic utilization (deployed / equity) exceeds a ceiling.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

import aiohttp

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    MARGIN_LEVEL_HALT_BELOW,
    SYNTHETIC_DEPLOYMENT_HALT_ABOVE,
)

# Signed SAPI lives on api root, not under /api/v3
BINANCE_SAPI_ROOT = "https://api.binance.com"

log = logging.getLogger(__name__)


def _sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


async def fetch_margin_account_snapshot(
    session: aiohttp.ClientSession,
) -> dict[str, Any] | None:
    """
    GET /sapi/v1/margin/account — requires API key + signature.

    Returns parsed JSON or None on failure / missing keys.
    """
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return None
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000,
    }
    q = urlencode(params)
    sig = _sign(q, BINANCE_API_SECRET)
    url = f"{BINANCE_SAPI_ROOT}/sapi/v1/margin/account?{q}&signature={sig}"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=12)) as r:
            if r.status != 200:
                txt = await r.text()
                log.warning("margin/account HTTP %s: %s", r.status, txt[:200])
                return None
            return await r.json()
    except Exception as exc:
        log.warning("margin/account error: %s", exc)
        return None


def _parse_margin_level(data: dict[str, Any]) -> float | None:
    raw = data.get("marginLevel")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _estimate_available_usdt(data: dict[str, Any]) -> float | None:
    """Sum free USDT in cross margin account (best-effort)."""
    assets = data.get("userAssets") or data.get("assets") or []
    total = 0.0
    for a in assets:
        if str(a.get("asset", "")).upper() == "USDT":
            try:
                total += float(a.get("free", 0) or 0)
            except (TypeError, ValueError):
                pass
    return total if total > 0 else None


def evaluate_margin_health(
    *,
    live_json: dict[str, Any] | None,
    cash: float,
    total_equity: float,
) -> dict[str, Any]:
    """
    Unified margin health record for bot + dashboard.

    `halt_new_entries` is True when we must not add exposure.
    """
    halt = False
    reason: str | None = None
    source = "synthetic"
    margin_level: float | None = None
    available: float | None = None
    utilization_pct: float | None = None

    if live_json and isinstance(live_json, dict):
        source = "binance_sapi"
        margin_level = _parse_margin_level(live_json)
        available = _estimate_available_usdt(live_json)
        if margin_level is not None and margin_level < MARGIN_LEVEL_HALT_BELOW:
            halt = True
            reason = f"margin_level={margin_level:.4f}<{MARGIN_LEVEL_HALT_BELOW}"

    if total_equity > 1e-9:
        deployed = max(0.0, total_equity - cash)
        utilization_pct = 100.0 * deployed / total_equity
        if utilization_pct / 100.0 >= SYNTHETIC_DEPLOYMENT_HALT_ABOVE:
            halt = True
            reason = reason or f"synthetic_util={utilization_pct:.1f}%>={SYNTHETIC_DEPLOYMENT_HALT_ABOVE*100:.0f}%"

    return {
        "halt_new_entries": halt,
        "halt_reason": reason,
        "source": source,
        "margin_level": margin_level,
        "available_margin_usdt": available,
        "utilization_pct": utilization_pct,
        "cash": round(cash, 2),
        "total_equity": round(total_equity, 2),
    }
