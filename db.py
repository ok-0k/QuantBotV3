"""
db.py — SQLite data layer with WAL pragmas.

SHORTING ADDITIONS vs previous version:

  Schema changes (positions table):
    side TEXT DEFAULT 'long'
      — 'long' for buy-first positions (existing behaviour, default)
      — 'short' for sell-first positions (new)
      All existing rows get 'long' automatically via the migration command.

    margin_reserved REAL DEFAULT 0
      — Cash held as collateral for a short position.
      — On open_short: this amount is deducted from cash and stored here.
      — On close_short: returned to cash + or - the PnL.
      — For longs this is always 0 (longs use cash directly, not margin).

  Schema changes (trades table):
    side TEXT DEFAULT 'long'
      — Records whether each trade was the long or short side.
      — Allows trade history to clearly distinguish long buys/sells from
        short entries/covers.

  New functions:
    open_short()           — Opens a new short position with margin reservation.
    close_short()          — Covers a short, returns margin, books PnL.
    get_short_position()   — Returns short position for a symbol (if any).
    get_all_short_positions() — Returns all open short positions.
    open_short_count()     — Count of currently open shorts (for MAX check).

  Modified functions:
    get_position()         — Default dict now includes side and margin_reserved.
    get_all_positions()    — Now queries WHERE shares > 0 OR margin_reserved > 0
                             so shorts (which have 0 shares) appear in position lists.
    open_position()        — Explicitly marks side='long'; margin_reserved stays 0.
    close_position()       — Resets side to 'long', margin_reserved to 0.
    log_trade()            — Accepts optional 'side' key in trade dict.
    init_db()              — Adds migration blocks for both new columns.

  PnL calculation for shorts:
    Long PnL  = (exit_price - entry_price) × shares   [profit when price rises]
    Short PnL = (entry_price - exit_price) × shares   [profit when price falls]
    This is handled in close_short() not here in db.py — db.py just stores the
    PnL that bot.py calculates and passes in.

UNCHANGED from previous version:
  All long-side functions behave identically. Existing open positions are
  unaffected. The 'side' column defaults to 'long' for all existing rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any

from config import DB_PATH, STARTING_CASH, SYMBOLS

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size=30000000000;")
        conn.execute("PRAGMA cache_size=-65536;")
        _local.conn = conn
    return _local.conn


@contextmanager
def get_db():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol  TEXT    NOT NULL,
                ts      INTEGER NOT NULL,
                open    REAL    NOT NULL,
                high    REAL    NOT NULL,
                low     REAL    NOT NULL,
                close   REAL    NOT NULL,
                volume  REAL    NOT NULL,
                PRIMARY KEY (symbol, ts)
            );
            CREATE INDEX IF NOT EXISTS idx_candles_sym_ts ON candles(symbol, ts DESC);

            CREATE TABLE IF NOT EXISTS portfolio (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                side        TEXT    DEFAULT 'long',
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

            CREATE TABLE IF NOT EXISTS positions (
                symbol          TEXT PRIMARY KEY,
                side            TEXT    DEFAULT 'long',
                shares          REAL    DEFAULT 0,
                avg_cost        REAL    DEFAULT 0,
                strategy        TEXT    DEFAULT '',
                stop_price      REAL    DEFAULT 0,
                tp_price        REAL    DEFAULT 0,
                candle_count    INTEGER DEFAULT 0,
                on_fire         INTEGER DEFAULT 0,
                opened_ts       TEXT,
                entry_state     TEXT    DEFAULT NULL,
                margin_reserved REAL    DEFAULT 0,
                mfe_price       REAL    DEFAULT NULL,
                mae_price       REAL    DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS equity_curve (
                ts     TEXT NOT NULL,
                equity REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts DESC);

            CREATE TABLE IF NOT EXISTS ml_cache (
                symbol   TEXT PRIMARY KEY,
                features TEXT NOT NULL,
                prob     REAL DEFAULT 0.5,
                updated  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rl_experience (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                symbol     TEXT    NOT NULL,
                state      TEXT    NOT NULL,
                action     REAL    NOT NULL,
                reward     REAL    NOT NULL,
                next_state TEXT    NOT NULL,
                done       INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS brain_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- V4 order-intent journal. Written BEFORE any fill is attempted so
            -- a crash mid-order leaves an auditable non-terminal row that
            -- startup reconciliation can resolve. client_order_id is
            -- deterministic per (symbol, action, candle) which makes entry
            -- submission idempotent: retries and duplicate signals collapse
            -- onto the same row instead of creating a second order.
            CREATE TABLE IF NOT EXISTS orders (
                client_order_id TEXT PRIMARY KEY,
                ts_created      TEXT NOT NULL,
                ts_updated      TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                action          TEXT NOT NULL,
                side            TEXT NOT NULL,
                candle_ts       INTEGER,
                req_qty         REAL,
                req_price       REAL,
                filled_qty      REAL DEFAULT 0,
                avg_fill_price  REAL DEFAULT 0,
                state           TEXT NOT NULL,
                exchange_order_id TEXT,
                mode            TEXT NOT NULL DEFAULT 'paper',
                note            TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_orders_sym_state ON orders(symbol, state);
        """)

        # ── SAFE MIGRATIONS (idempotent — run every startup) ──────────────────
        # Each block checks whether the column exists before adding it.
        # Safe to run against the live database repeatedly.
        existing_pos = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
        existing_trd = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}

        # positions table migrations
        if "entry_state" not in existing_pos:
            conn.execute("ALTER TABLE positions ADD COLUMN entry_state TEXT DEFAULT NULL")
        if "side" not in existing_pos:
            conn.execute("ALTER TABLE positions ADD COLUMN side TEXT DEFAULT 'long'")
        if "margin_reserved" not in existing_pos:
            conn.execute("ALTER TABLE positions ADD COLUMN margin_reserved REAL DEFAULT 0")
        if "mfe_price" not in existing_pos:
            conn.execute("ALTER TABLE positions ADD COLUMN mfe_price REAL DEFAULT NULL")
        if "mae_price" not in existing_pos:
            conn.execute("ALTER TABLE positions ADD COLUMN mae_price REAL DEFAULT NULL")

        # trades table migrations
        if "side" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN side TEXT DEFAULT 'long'")
        if "fee_total" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN fee_total REAL")
        if "net_pnl" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN net_pnl REAL")
        if "gross_pnl" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN gross_pnl REAL")
        if "allocated_equity_pct" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN allocated_equity_pct REAL")
        if "max_unrealized_pnl" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN max_unrealized_pnl REAL")
        if "min_unrealized_pnl" not in existing_trd:
            conn.execute("ALTER TABLE trades ADD COLUMN min_unrealized_pnl REAL")

    _seed_portfolio()


def _seed_portfolio() -> None:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM portfolio WHERE key='cash'").fetchone()
        if row is None:
            conn.execute("INSERT INTO portfolio VALUES ('cash', ?)", (str(STARTING_CASH),))
            conn.execute("INSERT INTO portfolio VALUES ('total_trades', '0')")
            conn.execute("INSERT INTO portfolio VALUES ('realised_pnl', '0.0')")
            for sym in SYMBOLS:
                conn.execute("INSERT OR IGNORE INTO positions(symbol) VALUES (?)", (sym,))


# ─────────────────────────────────────────────────────────────────────────────
# CANDLES
# ─────────────────────────────────────────────────────────────────────────────

def upsert_candle(symbol: str, candle: dict) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO candles (symbol,ts,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?)",
            (symbol, int(candle["time"]), candle["open"], candle["high"],
             candle["low"], candle["close"], candle["volume"]),
        )


def upsert_candles_bulk(symbol: str, candles: list[dict]) -> None:
    rows = [(symbol, int(c["time"]), c["open"], c["high"],
             c["low"], c["close"], c["volume"]) for c in candles]
    with get_db() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO candles (symbol,ts,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?)", rows,
        )


def get_candles(symbol: str, limit: int = 300) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT ts AS time, open, high, low, close, volume "
            "FROM candles WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def get_cash() -> float:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM portfolio WHERE key='cash'").fetchone()
    return float(row[0]) if row else STARTING_CASH


def set_cash(value: float) -> None:
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES ('cash',?)", (str(value),))


def get_portfolio_stat(key: str, default: Any = None) -> Any:
    with get_db() as conn:
        row = conn.execute("SELECT value FROM portfolio WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_portfolio_stat(key: str, value: Any) -> None:
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES (?,?)", (key, str(value)))


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS — READ
# ─────────────────────────────────────────────────────────────────────────────

def get_position(symbol: str) -> dict:
    """Return position for symbol regardless of side. Includes short positions."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
    if row:
        return dict(row)
    return {
        "symbol": symbol, "side": "long", "shares": 0.0, "avg_cost": 0.0,
        "strategy": "", "stop_price": 0.0, "tp_price": 0.0,
        "candle_count": 0, "on_fire": 0, "entry_state": None,
        "margin_reserved": 0.0,
    }


def get_all_positions() -> list[dict]:
    """
    Return all active positions — both longs (shares > 0) and shorts
    (margin_reserved > 0, shares is used to track the shorted quantity).
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE shares > 0 OR margin_reserved > 0"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_long_positions() -> list[dict]:
    """Return only open long positions."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE side='long' AND shares > 0"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_short_positions() -> list[dict]:
    """Return only open short positions."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE side='short' AND margin_reserved > 0"
        ).fetchall()
    return [dict(r) for r in rows]


def get_short_position(symbol: str) -> dict | None:
    """Return the short position for a symbol, or None if not shorted."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol=? AND side='short' AND margin_reserved > 0",
            (symbol,),
        ).fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS — LONG (buy first, sell to close)
# ─────────────────────────────────────────────────────────────────────────────

def open_position(symbol: str, shares: float, avg_cost: float,
                  strategy: str, stop_price: float, tp_price: float,
                  on_fire: bool = False,
                  entry_state: list | None = None) -> None:
    """
    Open or average into a LONG position.

    Behaviour is identical to the previous version — existing open positions
    are averaged (weighted cost basis, tightest stop preserved).
    The side column is explicitly set to 'long'.
    margin_reserved is always 0 for longs (cash is deducted at buy time).
    """
    now = datetime.now(timezone.utc).isoformat()
    entry_state_json = json.dumps(entry_state) if entry_state is not None else None

    with get_db() as conn:
        existing = conn.execute(
            "SELECT shares, avg_cost, stop_price, side FROM positions WHERE symbol=?",
            (symbol,),
        ).fetchone()

        # Don't average into a short — that would be a conflicting position.
        # The bot layer should close the short first; we treat this as a new long.
        if existing and existing["shares"] > 0 and existing["side"] == "long":
            # ── POSITION AVERAGING ────────────────────────────────────────────
            old_shares   = existing["shares"]
            old_cost     = existing["avg_cost"]
            total_shares = old_shares + shares
            new_avg_cost = (old_shares * old_cost + shares * avg_cost) / total_shares
            new_stop     = max(existing["stop_price"], stop_price)

            conn.execute("""
                UPDATE positions
                SET shares=?, avg_cost=?, stop_price=?, tp_price=?, strategy=?,
                    on_fire=?, opened_ts=COALESCE(opened_ts, ?), side='long'
                WHERE symbol=?
            """, (total_shares, new_avg_cost, new_stop, tp_price, strategy,
                  int(on_fire), now, symbol))
        else:
            # ── NEW LONG POSITION ─────────────────────────────────────────────
            conn.execute("""
                INSERT OR REPLACE INTO positions
                    (symbol, side, shares, avg_cost, strategy, stop_price, tp_price,
                     candle_count, on_fire, opened_ts, entry_state, margin_reserved)
                VALUES (?,?,?,?,?,?,?,0,?,?,?,0)
            """, (symbol, "long", shares, avg_cost, strategy, stop_price, tp_price,
                  int(on_fire), now, entry_state_json))


def delete_position(symbol: str) -> None:
    """
    Hard-delete any position row for `symbol` regardless of its current state.
    Called by:
      • close_position / close_short after a legitimate trade closes (Fix 3).
      • exit_monitor zombie-guard when shares == 0 / None (Fix 2).
    Using DELETE instead of UPDATE prevents stale zero-share rows from leaking
    back into get_all_positions() if the WHERE filter ever changes.
    """
    with get_db() as conn:
        conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))


def close_position(symbol: str) -> None:
    """Close a long position — hard-delete the row so no zombie shell remains."""
    delete_position(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS — SHORT (sell first, buy back to close)
# ─────────────────────────────────────────────────────────────────────────────

def open_short(symbol: str, shares: float, entry_price: float,
               strategy: str, stop_price: float, tp_price: float,
               margin_reserved: float,
               on_fire: bool = False,
               entry_state: list | None = None) -> None:
    """
    Open a SHORT position.

    For a short:
      - shares     = number of units sold short (positive number)
      - entry_price = the price at which we sold short (avg_cost column)
      - stop_price = price ABOVE entry where we buy back at a loss
                     (entry_price + N × ATR, because short stops go up)
      - tp_price   = price BELOW entry where we buy back for profit
                     (entry_price - N × ATR)
      - margin_reserved = cash locked as collateral (deducted from cash by bot.py)

    Shorting the same symbol twice is NOT allowed — raises ValueError if an
    active short already exists for this symbol. The bot layer must close
    the existing short (close_short()) before opening a new one. Previously
    this was an unenforced comment: INSERT OR REPLACE would silently
    overwrite shares/avg_cost/stop_price/opened_ts, orphaning the original
    short's margin_reserved from cash forever (it was deducted but never
    tracked as refundable) and resetting candle_count/opened_ts, defeating
    the time-based exit protections.

    PnL when closing:
      profit = (entry_price - cover_price) × shares
      positive when cover_price < entry_price (price fell, as intended)
      negative when cover_price > entry_price (price rose, short squeeze)
    """
    now = datetime.now(timezone.utc).isoformat()
    entry_state_json = json.dumps(entry_state) if entry_state is not None else None

    with get_db() as conn:
        existing = conn.execute(
            "SELECT margin_reserved FROM positions WHERE symbol=? AND side='short'",
            (symbol,),
        ).fetchone()
        if existing and float(existing["margin_reserved"] or 0.0) > 0:
            raise ValueError(
                f"open_short: {symbol} already has an active short "
                f"(margin_reserved={existing['margin_reserved']}) — "
                f"close it via close_short() before opening a new one"
            )
        conn.execute("""
            INSERT OR REPLACE INTO positions
                (symbol, side, shares, avg_cost, strategy, stop_price, tp_price,
                 candle_count, on_fire, opened_ts, entry_state, margin_reserved)
            VALUES (?,?,?,?,?,?,?,0,?,?,?,?)
        """, (symbol, "short", shares, entry_price, strategy, stop_price, tp_price,
              int(on_fire), now, entry_state_json, margin_reserved))


def close_short(symbol: str) -> None:
    """
    Cover a short position — hard-delete the row so no zombie shell remains.
    Cash management is handled by bot.py before this call.
    """
    delete_position(symbol)


def reduce_position(symbol: str, qty_closed: float, margin_released: float = 0.0) -> None:
    """
    Partially close a position (V4 — only reachable on testnet partial fills;
    paper fills are always complete). Shrinks shares and, for shorts, the
    reserved margin. Stops/targets/entry stay as they were.
    """
    with get_db() as conn:
        conn.execute(
            "UPDATE positions SET shares = MAX(0, shares - ?),"
            " margin_reserved = MAX(0, margin_reserved - ?) WHERE symbol=?",
            (float(qty_closed), float(margin_released), symbol),
        )


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS — MFE / MAE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def update_mfe_mae(symbol: str, candle_high: float, candle_low: float, is_short: bool) -> None:
    """
    Ratchet Maximum Favorable Excursion (mfe_price) and Maximum Adverse Excursion
    (mae_price) prices for an open position based on the current candle extremes.

    Long  positions: mfe = highest high ever reached (best exit price),
                     mae = lowest low ever reached (worst drawdown price).
    Short positions: mfe = lowest low ever reached (best cover price),
                     mae = highest high ever reached (worst squeeze price).

    Only writes to DB when the candle moves beyond the prior recorded extreme,
    so most ticks produce no write (the SELECT check is the hot path).
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT mfe_price, mae_price FROM positions WHERE symbol=?",
            (symbol,),
        ).fetchone()
        if not row:
            return

        old_mfe = row[0]   # None on first tick
        old_mae = row[1]   # None on first tick

        if not is_short:
            new_mfe = candle_high if old_mfe is None else max(old_mfe, candle_high)
            new_mae = candle_low  if old_mae is None else min(old_mae, candle_low)
        else:
            new_mfe = candle_low  if old_mfe is None else min(old_mfe, candle_low)
            new_mae = candle_high if old_mae is None else max(old_mae, candle_high)

        # Skip the write if nothing changed (the common case after the first few bars)
        if new_mfe == old_mfe and new_mae == old_mae:
            return

        conn.execute(
            "UPDATE positions SET mfe_price=?, mae_price=? WHERE symbol=?",
            (new_mfe, new_mae, symbol),
        )


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS — TRAILING STOP
# ─────────────────────────────────────────────────────────────────────────────

def update_tp_price(symbol: str, new_tp: float) -> None:
    """Update take-profit level (used by ML time-decay exits)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE positions SET tp_price=? WHERE symbol=?",
            (new_tp, symbol),
        )


def update_stop_price(symbol: str, new_stop: float) -> None:
    """
    Ratchet the stop_price for an open position (long or short) to lock in profit.

    For a long  : call only when new_stop > current stop_price (stop moves up).
    For a short : call only when new_stop < current stop_price (stop moves down).

    The caller (bot.py) is responsible for the directional guard — this function
    just writes whatever value it receives, so it stays side-agnostic and simple.
    """
    with get_db() as conn:
        conn.execute(
            "UPDATE positions SET stop_price=? WHERE symbol=?",
            (new_stop, symbol),
        )


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS — COUNTS
# ─────────────────────────────────────────────────────────────────────────────

def open_position_count() -> int:
    """Count of open long positions."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE side='long' AND shares > 0"
        ).fetchone()
    return row[0] if row else 0


def open_short_count() -> int:
    """Count of open short positions."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE side='short' AND margin_reserved > 0"
        ).fetchone()
    return row[0] if row else 0


def increment_candle_count(symbol: str) -> int:
    with get_db() as conn:
        conn.execute(
            "UPDATE positions SET candle_count=candle_count+1 WHERE symbol=?",
            (symbol,),
        )
        row = conn.execute(
            "SELECT candle_count FROM positions WHERE symbol=?", (symbol,)).fetchone()
    return row[0] if row else 0


def get_entry_state(symbol: str) -> list | None:
    """Retrieve persisted SAC entry state for RL transition reconstruction."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT entry_state FROM positions WHERE symbol=?", (symbol,)).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────

def log_trade(trade: dict) -> None:
    """
    Log a trade. Accepts optional 'side' key ('long' or 'short').
    Defaults to 'long' if not provided, preserving backward compatibility
    with all existing log_trade() call sites in bot.py.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO trades
                (ts,symbol,action,side,strategy,regime,price,exec_price,
                 shares,trade_value,pnl,slippage,on_fire,status,reason,
                 fee_total,net_pnl,gross_pnl,allocated_equity_pct,
                 max_unrealized_pnl,min_unrealized_pnl)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade.get("timestamp", now),
            trade.get("symbol"),
            trade.get("action"),
            trade.get("side", "long"),
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
            trade.get("fee_total"),
            trade.get("net_pnl"),
            trade.get("gross_pnl"),
            trade.get("allocated_equity_pct"),
            trade.get("max_unrealized_pnl"),
            trade.get("min_unrealized_pnl"),
        ))


def get_recent_trades(limit: int = 50) -> list[dict]:
    """Return the most recent *closed* trades (exit legs only, net_pnl populated).

    Entry-leg rows (buy / short) are excluded because they have NULL net_pnl and
    would appear as blank duplicate rows in the Trade History table.
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='filled' AND net_pnl IS NOT NULL"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_filled_trade_count() -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='filled'").fetchone()
    return row[0] if row else 0


def get_cash_curve_from_trades(limit: int = 200) -> list[dict]:
    """Reconstruct a cash time-series from closed trade PnL.

    Works backwards from the current cash balance, undoing each trade's
    net_pnl to recover what cash was at that moment.  Returns points in
    chronological order so Chart.js gets an ascending time series.
    """
    current_cash = get_cash()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT ts, net_pnl FROM trades"
            " WHERE status='filled' AND net_pnl IS NOT NULL"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return [{"time": now_str, "cash": round(current_cash, 2)},
                {"time": now_str, "cash": round(current_cash, 2)}]
    points = []
    running = current_cash
    for r in rows:
        ts = str(r["ts"])[:19].replace("T", " ")
        points.append({"time": ts, "cash": round(running, 2)})
        running -= float(r["net_pnl"] or 0.0)
    points.reverse()  # oldest first
    return points


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY CURVE
# ─────────────────────────────────────────────────────────────────────────────

def get_equity_rebase() -> tuple[float, str] | tuple[None, None]:
    """
    Return (baseline, ts) set by rebase_equity_baseline.py, or (None, None) if
    no rebase has ever been declared. Deliberately separate from
    STARTING_CASH: STARTING_CASH keeps meaning "what a fresh install seeds
    cash with" and "the reference point for the cash-invariant check";
    this is purely a dashboard display baseline for return_pct / peak-equity.
    """
    with get_db() as conn:
        b = conn.execute("SELECT value FROM portfolio WHERE key='equity_rebase_baseline'").fetchone()
        t = conn.execute("SELECT value FROM portfolio WHERE key='equity_rebase_ts'").fetchone()
    if b is None or t is None:
        return None, None
    try:
        return float(b[0]), str(t[0])
    except (TypeError, ValueError):
        return None, None


def set_equity_rebase(baseline: float, ts: str) -> None:
    """Declare a new performance-display epoch. See rebase_equity_baseline.py."""
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES ('equity_rebase_baseline', ?)", (str(baseline),))
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES ('equity_rebase_ts', ?)", (ts,))


def record_equity(equity: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    eq = float(equity)
    baseline, _ = get_equity_rebase()
    ref = baseline if baseline is not None and baseline > 0 else float(STARTING_CASH)
    ret = ((eq - ref) / ref) * 100.0
    with get_db() as conn:
        conn.execute("INSERT INTO equity_curve (ts, equity) VALUES (?,?)", (now, eq))
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES ('current_equity', ?)", (str(eq),))
        conn.execute("INSERT OR REPLACE INTO portfolio VALUES ('return_pct', ?)", (str(ret),))


def get_equity_curve(limit: int = 500) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT ts, equity FROM equity_curve ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    # Return 19-char timestamps (with seconds) so _utc_to_van can parse them.
    # Previously [:16] stripped seconds, causing _utc_to_van to silently return
    # raw UTC strings (all its format strings require %H:%M:%S).
    return [{"time": r["ts"][:19].replace("T", " "), "equity": r["equity"]}
            for r in reversed(rows)]


def get_trades_last_7_days() -> list[dict]:
    """Return all *closed* trades from the last 7 days (exit legs only, no row cap).

    The net_pnl IS NOT NULL guard mirrors get_recent_trades: entry-leg rows
    (buy / short) have NULL net_pnl and must be excluded from analytics so that
    win-rate, fee-drag, and weekly-breakdown calculations are not inflated by
    duplicate entry rows.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades"
            " WHERE status='filled' AND net_pnl IS NOT NULL AND ts >= ?"
            " ORDER BY id DESC",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# ML CACHE
# ─────────────────────────────────────────────────────────────────────────────

def save_ml_cache(symbol: str, features: list, prob: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ml_cache (symbol,features,prob,updated) VALUES (?,?,?,?)",
            (symbol, json.dumps(features), prob, now),
        )


def get_ml_prob(symbol: str) -> float:
    with get_db() as conn:
        row = conn.execute(
            "SELECT prob FROM ml_cache WHERE symbol=?", (symbol,)).fetchone()
    return float(row[0]) if row else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# RL EXPERIENCE
# ─────────────────────────────────────────────────────────────────────────────

def log_rl_experience(symbol: str, state: list, action: float,
                      reward: float, next_state: list, done: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO rl_experience (ts,symbol,state,action,reward,next_state,done)
            VALUES (?,?,?,?,?,?,?)
        """, (now, symbol, json.dumps(state), action, reward,
              json.dumps(next_state), int(done)))


def get_rl_experience(limit: int = 10000) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT state,action,reward,next_state,done "
            "FROM rl_experience ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# BRAIN STATE
# ─────────────────────────────────────────────────────────────────────────────

def save_brain_key(key: str, value: Any) -> None:
    """
    Persist an arbitrary Python value under `key` in the brain_state table.

    Fix 1 — three hardening layers vs the original:
      • Explicit column names in INSERT OR REPLACE so positional ambiguity can
        never silently swap key↔value if the schema ever gains columns.
      • Explicit conn.commit() AFTER the context-manager commit as a belt-and-
        suspenders WAL flush.  WAL mode writes to the journal first; the explicit
        commit checkpoints the page to the main DB file so the row survives
        a hard process kill between context-manager exit and OS buffer flush.
      • Separate try/except on the commit so a flush failure is logged rather than
        silently swallowed.
    """
    serialised = json.dumps(value)
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO brain_state (key, value) VALUES (?, ?)",
            (key, serialised),
        )
        # Belt-and-suspenders: force WAL checkpoint synchronously.
        try:
            conn.commit()
        except Exception:
            pass   # get_db() already rolled back; caller will see the exception


def load_brain_key(key: str, default: Any = None) -> Any:
    """Load a value from brain_state; returns `default` on missing key or bad JSON."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM brain_state WHERE key=?", (key,)).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            return default
    return default


# ─────────────────────────────────────────────────────────────────────────────
# ORDER JOURNAL (V4)
# ─────────────────────────────────────────────────────────────────────────────
# State machine:
#   PENDING_NEW -> FILLED | PARTIALLY_FILLED | REJECTED | FAILED | ORPHANED
# Terminal states: FILLED, REJECTED, FAILED, ORPHANED.
# PARTIALLY_FILLED is non-terminal (testnet reconciliation may complete it).

ORDER_TERMINAL_STATES = ("FILLED", "REJECTED", "FAILED", "ORPHANED")


def try_create_order(client_order_id: str, symbol: str, action: str, side: str,
                     candle_ts: int | None, req_qty: float, req_price: float,
                     mode: str) -> bool:
    """
    Atomically claim an order intent. Returns True if this call created the
    row (caller owns the order), False if the client_order_id already exists —
    i.e. a duplicate signal or a retry raced us. INSERT OR IGNORE on the
    PRIMARY KEY makes this race-safe across threads sharing the DB.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO orders
                (client_order_id, ts_created, ts_updated, symbol, action, side,
                 candle_ts, req_qty, req_price, state, mode)
            VALUES (?,?,?,?,?,?,?,?,?, 'PENDING_NEW', ?)
            """,
            (client_order_id, now, now, symbol, action, side,
             candle_ts, req_qty, req_price, mode),
        )
        return cur.rowcount == 1


def update_order(client_order_id: str, *, state: str,
                 filled_qty: float | None = None,
                 avg_fill_price: float | None = None,
                 exchange_order_id: str | None = None,
                 note: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sets = ["state=?", "ts_updated=?"]
    vals: list[Any] = [state, now]
    if filled_qty is not None:
        sets.append("filled_qty=?")
        vals.append(float(filled_qty))
    if avg_fill_price is not None:
        sets.append("avg_fill_price=?")
        vals.append(float(avg_fill_price))
    if exchange_order_id is not None:
        sets.append("exchange_order_id=?")
        vals.append(str(exchange_order_id))
    if note is not None:
        sets.append("note=?")
        vals.append(note)
    vals.append(client_order_id)
    with get_db() as conn:
        conn.execute(f"UPDATE orders SET {', '.join(sets)} WHERE client_order_id=?", vals)


def get_order(client_order_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
    return dict(row) if row else None


def get_open_orders() -> list[dict]:
    """All journal rows in non-terminal states (candidates for reconciliation)."""
    placeholders = ",".join("?" for _ in ORDER_TERMINAL_STATES)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM orders WHERE state NOT IN ({placeholders})"
            " ORDER BY ts_created",
            ORDER_TERMINAL_STATES,
        ).fetchall()
    return [dict(r) for r in rows]