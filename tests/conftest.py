"""
Test harness isolation.

CRITICAL ORDERING: config.py resolves TRADING_DATA_DIR / TRADING_LOG_DIR at
import time, so the environment must be prepared before any project module is
imported. pytest imports conftest.py first, which is what makes this safe.

Every test session runs against a throwaway SQLite DB in a temp directory —
the live DB at /home/admin/trading_data is never touched. Discord alerts are
disabled via env and belt-and-suspenders monkeypatching.
"""

import os
import sys
import tempfile
from pathlib import Path

_SESSION_TMP = tempfile.mkdtemp(prefix="quantbot-test-")
os.environ["TRADING_DATA_DIR"] = str(Path(_SESSION_TMP) / "data")
os.environ["TRADING_LOG_DIR"] = str(Path(_SESSION_TMP) / "logs")
os.environ["TRADING_ENV_FILE"] = str(Path(_SESSION_TMP) / "no-such-env-file")
os.environ["DISCORD_WEBHOOK_URL"] = ""
os.environ["EXECUTION_MODE"] = "paper"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402

assert str(config.DB_PATH).startswith(_SESSION_TMP), (
    f"Test DB escaped the sandbox: {config.DB_PATH}"
)

import db  # noqa: E402


@pytest.fixture()
def clean_db():
    """Fresh schema + seeded portfolio; wipes all rows between tests."""
    db.init_db()
    with db.get_db() as conn:
        for table in (
            "positions", "trades", "equity_curve", "rl_experience",
            "ml_cache", "brain_state", "portfolio", "candles", "orders",
        ):
            conn.execute(f"DELETE FROM {table}")
    db._seed_portfolio()
    return db


@pytest.fixture(autouse=True)
def _no_discord(monkeypatch):
    """No network alerts, ever, regardless of env handling in the module."""
    try:
        import bot
        monkeypatch.setattr(bot, "DISCORD_WEBHOOK_URL", "", raising=False)
        monkeypatch.setattr(bot, "alert_sniper_shot", lambda *a, **k: None, raising=False)
    except ImportError:
        pass
    yield


def make_candles(
    n: int = 120,
    start_price: float = 100.0,
    drift: float = 0.0,
    amplitude: float = 0.5,
    volume: float = 1000.0,
    start_time_ms: int | None = None,
    interval_ms: int = 60_000,
) -> list[dict]:
    """
    Deterministic synthetic OHLCV series (prices are RNG-free; reproducible).

    start_time_ms=None (default) stamps the series so the LAST candle opened
    one interval ago — i.e. FRESH data that passes the V4 stale-entry gate.
    Pass an old fixed epoch to simulate a dead feed.

    drift: per-bar close-to-close delta. amplitude: high/low spread around close.
    """
    if start_time_ms is None:
        import time as _time
        start_time_ms = int(_time.time() * 1000) - n * interval_ms
    candles = []
    price = start_price
    for i in range(n):
        close = price + drift
        wiggle = amplitude * (0.5 + 0.5 * ((i * 37) % 100) / 100.0)
        candles.append({
            "time": start_time_ms + i * interval_ms,
            "open": round(price, 8),
            "high": round(max(price, close) + wiggle, 8),
            "low": round(min(price, close) - wiggle, 8),
            "close": round(close, 8),
            "volume": volume,
        })
        price = close
    return candles


def make_correlated_walk(
    n: int = 130,
    seed: int = 1,
    start_price: float = 100.0,
    start_time_ms: int = 1_700_000_000_000,
) -> list[dict]:
    """Deterministic pseudo-random walk (LCG) — same seed => identical series."""
    candles = []
    price = start_price
    state = seed
    for i in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        step = ((state / (2 ** 31)) - 0.5) * 0.8
        close = max(1.0, price + step)
        candles.append({
            "time": start_time_ms + i * 60_000,
            "open": round(price, 8),
            "high": round(max(price, close) + 0.1, 8),
            "low": round(min(price, close) - 0.1, 8),
            "close": round(close, 8),
            "volume": 1000.0,
        })
        price = close
    return candles
