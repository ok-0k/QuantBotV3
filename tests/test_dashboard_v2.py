"""
Dashboard v2 (audit round 5): server-side parts of the dashboard fixes --
auth on non-ASCII tokens, ledger tile, breaker drawdown, bot liveness,
lifetime trade count, experiment per-symbol-day metric, cleared-cost
threshold.
"""

import asyncio

import pytest
from starlette.requests import Request

import dashboard


def _request(query: bytes = b"", headers: list | None = None) -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "query_string": query,
                    "headers": headers or [], "server": ("t", 80), "scheme": "http"})


def _call_middleware(req: Request):
    async def call_next(_req):
        return dashboard.JSONResponse({"ok": True})
    return asyncio.run(dashboard._require_token(req, call_next))


def test_auth_non_ascii_token_is_401_not_500(monkeypatch):
    monkeypatch.setattr(dashboard, "_AUTH_TOKEN", "s3cret")
    assert _call_middleware(_request(b"token=%C3%A9")).status_code == 401
    assert _call_middleware(_request(headers=[(b"authorization", "Bearer é".encode())])).status_code == 401
    assert _call_middleware(_request(b"token=s3cret")).status_code == 200


def test_auth_non_ascii_configured_token(monkeypatch):
    monkeypatch.setattr(dashboard, "_AUTH_TOKEN", "pässwörd")
    assert _call_middleware(_request(b"token=p%C3%A4ssw%C3%B6rd")).status_code == 200
    assert _call_middleware(_request(b"token=nope")).status_code == 401
