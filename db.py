"""
db.py — In-memory-mapped SQLite database layer.

Replaces all JSON file I/O.  Every PRAGMA is documented with its exact
Pi 5 impact per the architectural blueprint.

Thread safety: WAL mode allows the async bot to write while the
FastAPI dashboard reads — zero blocking between them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import DB_PATH, STARTING_CASH, SYMBOLS

# One connection object per OS thread (sqlite3 requires this in WAL mode)
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return thread-local SQLite connection with all performance PRAGMAs set."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # ── MANDATORY PRAGMAS for Raspberry Pi 5 edge deployment ─────────────
        # WAL: readers never block writers, writers never block readers.
        # Critical for the async bot (writer) + FastAPI dashboard (reader).
        conn.execute("PRAGMA journal_mode=WAL;")

        # NORMAL: skip full filesystem sync on every commit; OS cache handles it.
        # Increases write throughput to ~80,000 inserts/sec on ARM64.
        # Risk: last transaction lost on catastrophic power failure only.
        conn.execute("PRAGMA synchronous=NORMAL;")

        # MEMORY: temp tables, sort buffers, and indices live in RAM.
        # Eliminates disk I/O for complex queries; protects the SD card.
        conn.execute("PRAGMA temp_store=MEMORY;")

        # MMAP: memory-map 30 GB of the database file into address space.
        # OS page cache handles reads directly — bypasses read() syscalls.
        conn.execute("PRAGMA mmap_size=30000000000;")

        # 64 MB page cache — keeps hot data in RAM on the 8 GB Pi 5.
        conn.execute("PRAGMA cache_size=-65536;")

        _local.conn = conn
    return _local.conn


@contextmanager
def get_db():
    """Context manager yielding a connection; commits on success, rolls back on error."""
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not exist. Idempotent."""
    with get_db() as conn:
        conn.executescript("""
            -- OHLCV candle store — circular buffer via timestamp keying
            CREATE TABLE IF NOT EXISTS candles (
                symbol      TEXT    NOT NULL,
                ts          INTEGER NOT NULL,   -- Unix ms
                open        REAL    NOT NULL,
                high        REAL    NOT NULL,
                low         REAL    NOT NULL,
                close       REAL    NOT NULL,
                volume      REAL    NOT NULL,
                PRIMARY KEY (symbol, ts)
            );
            CREATE INDEX IF NOT EXISTS idx_candles_sym_ts ON candles(symbol, ts DESC);

            -- Portfolio state (single-row JSON store for simplicity)
            CREATE TABLE IF NOT EXISTS portfolio (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- All executed (filled) and attempted trades
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                strategy    TEXT,
                regime      TEXT,
                price       REAL,
                exec_price  REAL,
                shares      REAL,
                trade_value REAL,
                pnl         REAL,
                slippage    REAL,
                on_fire     INTEGER DEFAULT 0,
                status      TEXT    NOT NULL,
                reason      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_sym ON trades(symbol, ts DESC);

            -- Open positions
            CREATE TABLE IF NOT EXISTS positions (
                symbol       TEXT PRIMARY KEY,
                shares       REAL    DEFAULT 0,
                avg_cost     REAL    DEFAULT 0,
                strategy     TEXT    DEFAULT '',
                stop_price   REAL    DEFAULT 0,
                tp_price     REAL    DEFAULT 0,
                candle_count INTEGER DEFAULT 0,
                on_fire      INTEGER DEFAULT 0,
                opened_ts    TEXT
            );

            -- Equity curve snapshots (every scan)
            CREATE TABLE IF NOT EXISTS equity_curve (
                ts     TEXT NOT NULL,
                equity REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts DESC);

            -- ML feature cache (last computed features per symbol)
            CREATE TABLE IF NOT EXISTS ml_cache (
                symbol   TEXT PRIMARY KEY,
                features TEXT NOT NULL,          -- JSON
                prob     REAL DEFAULT 0.5,
                updated  TEXT NOT NULL
            );

            -- RL experience buffer for offline training export
            CREATE TABLE IF NOT EXISTS rl_experience (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT    NOT NULL,
                symbol  TEXT    NOT NULL,
                state   TEXT    NOT NULL,        -- JSON array
                action  REAL    NOT NULL,
                reward  REAL    NOT NULL,
                next_state TEXT NOT NULL,
                done    INTEGER DEFAULT 0
            );

            -- Brain / strategy state (persisted as JSON blob)
            CREATE TABLE IF NOT EXISTS brain_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    # Seed portfolio if empty
    _seed_portfolio()


def _seed_portfolio() -> None:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM portfolio WHERE key='cash'").fetchone()
        if row is None:
            conn.execute("INSERT INTO portfolio VALUES ('cash', ?)",
                         (str(STARTING_CASH),))
            conn.execute("INSERT INTO portfolio VALUES ('total_trades', '0')")
            conn.execute("INSERT INTO portfolio VALUES ('realised_pnl', '0.0')")
            # Initialise positions table for all symbols
            for sym in SYMBOLS:
                conn.execute("""
                    INSERT OR IGNORE INTO positions(symbol) VALUES (?)
                """, (sym,))


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def upsert_candle(symbol: str, candle: dict) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO candles
                (symbol, ts, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
        """, (symbol, int(candle["time"]), candle["open"], candle["high"],
              candle["low"], candle["close"], candle["volume"]))


def upsert_candles_bulk(symbol: str, candles: list[dict]) -> None:
    """Bulk upsert — used during startup history fetch."""
    rows = [(symbol, int(c["time"]), c["open"], c["high"],
             c["low"], c["close"], c["volume"]) for c in candles]
    with get_db() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO candles
                (symbol, ts, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
        """, rows)


def get_candles(symbol: str, limit: int = 300) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT ts AS time, open, high, low, close, volume
            FROM candles WHERE symbol=?
            ORDER BY ts DESC LIMIT ?
        """, (symbol, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_candle_count(symbol: str) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol=?", (symbol,)).fetchone()
    return row[0] if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_cash() -> float:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM portfolio WHERE key='cash'").fetchone()
    return float(row[0]) if row else STARTING_CASH


def set_cash(value: float) -> None:
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES ('cash', ?)",
                     (str(value),))


def get_portfolio_stat(key: str, default: Any = None) -> Any:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM portfolio WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_portfolio_stat(key: str, value: Any) -> None:
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES (?,?)",
                     (key, str(value)))


# ─────────────────────────────────────────────────────────────────────────────
# POSITION OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_position(symbol: str) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
    if row:
        return dict(row)
    return {"symbol": symbol, "shares": 0.0, "avg_cost": 0.0,
            "strategy": "", "stop_price": 0.0, "tp_price": 0.0,
            "candle_count": 0, "on_fire": 0}


def get_all_positions() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE shares > 0").fetchall()
    return [dict(r) for r in rows]


def open_position(symbol: str, shares: float, avg_cost: float,
                  strategy: str, stop_price: float, tp_price: float,
                  on_fire: bool = False) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO positions
                (symbol, shares, avg_cost, strategy, stop_price, tp_price,
                 candle_count, on_fire, opened_ts)
            VALUES (?,?,?,?,?,?,0,?,?)
        """, (symbol, shares, avg_cost, strategy, stop_price, tp_price,
              int(on_fire), now))


def close_position(symbol: str) -> None:
    with get_db() as conn:
        conn.execute("""
            UPDATE positions
            SET shares=0, avg_cost=0, strategy='', stop_price=0,
                tp_price=0, candle_count=0, on_fire=0, opened_ts=NULL
            WHERE symbol=?
        """, (symbol,))


def increment_candle_count(symbol: str) -> int:
    with get_db() as conn:
        conn.execute("""
            UPDATE positions SET candle_count = candle_count + 1
            WHERE symbol=?
        """, (symbol,))
        row = conn.execute(
            "SELECT candle_count FROM positions WHERE symbol=?",
            (symbol,)).fetchone()
    return row[0] if row else 0


def open_position_count() -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE shares > 0").fetchone()
    return row[0] if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────

def log_trade(trade: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO trades
                (ts, symbol, action, strategy, regime, price, exec_price,
                 shares, trade_value, pnl, slippage, on_fire, status, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade.get("timestamp", now),
            trade.get("symbol"),
            trade.get("action"),
            trade.get("strategy"),
            trade.get("regime"),
            trade.get("price"),
            trade.get("exec_price"),
            trade.get("shares"),
            trade.get("trade_value"),
            trade.get("pnl"),
            trade.get("slippage"),
            int(trade.get("on_fire", False)),
            trade.get("status", "skipped"),
            trade.get("reason"),
        ))


def get_recent_trades(limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM trades WHERE status='filled'
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_filled_trade_count() -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='filled'").fetchone()
    return row[0] if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY CURVE
# ─────────────────────────────────────────────────────────────────────────────

def record_equity(equity: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO equity_curve (ts, equity) VALUES (?,?)", (now, equity))


def get_equity_curve(limit: int = 500) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT ts, equity FROM equity_curve
            ORDER BY rowid DESC LIMIT ?
        """, (limit,)).fetchall()
    return [{"time": r["ts"][:16].replace("T", " "), "equity": r["equity"]}
            for r in reversed(rows)]


# ─────────────────────────────────────────────────────────────────────────────
# ML CACHE
# ─────────────────────────────────────────────────────────────────────────────

def save_ml_cache(symbol: str, features: list[float], prob: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO ml_cache (symbol, features, prob, updated)
            VALUES (?,?,?,?)
        """, (symbol, json.dumps(features), prob, now))


def get_ml_prob(symbol: str) -> float:
    with get_db() as conn:
        row = conn.execute(
            "SELECT prob FROM ml_cache WHERE symbol=?", (symbol,)).fetchone()
    return float(row[0]) if row else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# RL EXPERIENCE BUFFER
# ─────────────────────────────────────────────────────────────────────────────

def log_rl_experience(symbol: str, state: list, action: float,
                      reward: float, next_state: list, done: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO rl_experience (ts, symbol, state, action, reward, next_state, done)
            VALUES (?,?,?,?,?,?,?)
        """, (now, symbol, json.dumps(state), action, reward,
              json.dumps(next_state), int(done)))


def get_rl_experience(limit: int = 10000) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT state, action, reward, next_state, done
            FROM rl_experience ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# BRAIN STATE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_brain_key(key: str, value: Any) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO brain_state VALUES (?,?)",
            (key, json.dumps(value)))


def load_brain_key(key: str, default: Any = None) -> Any:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM brain_state WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    return default
