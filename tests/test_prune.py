"""
prune_db.sh must never delete trade data (trades / rl_experience are the
learning record and lifetime history) while still trimming candles and
thinning (not erasing) old equity history.
"""

import gzip
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "prune_db.sh"

pytestmark = pytest.mark.skipif(shutil.which("sqlite3") is None, reason="sqlite3 CLI not installed")


def _make_db(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE candles (symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL,
                              close REAL, volume REAL, PRIMARY KEY (symbol, ts));
        CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT,
                             action TEXT, status TEXT);
        CREATE TABLE rl_experience (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT);
        CREATE TABLE equity_curve (ts TEXT, equity REAL);
    """)
    conn.executemany("INSERT INTO candles VALUES ('AAAUSDT',?,1,1,1,1,1)", [(i,) for i in range(700)])
    conn.executemany("INSERT INTO trades (ts,symbol,action,status) VALUES (?,?,?,?)",
                     [(str(i), "AAAUSDT", "cover", "filled" if i % 3 else "skipped") for i in range(3000)])
    conn.executemany("INSERT INTO rl_experience (ts) VALUES (?)", [(str(i),) for i in range(2500)])
    # equity: 1 h of 30 s points 3 days ago (hour-aligned) + 1 h of 30 s points 1 h ago
    old0 = (datetime.now(timezone.utc) - timedelta(days=3)).replace(minute=0, second=0, microsecond=0)
    new0 = datetime.now(timezone.utc) - timedelta(hours=1)
    pts = [old0 + timedelta(seconds=30 * i) for i in range(120)]
    pts += [new0 + timedelta(seconds=30 * i) for i in range(120)]
    conn.executemany("INSERT INTO equity_curve VALUES (?,1)", [(t.isoformat(),) for t in pts])
    conn.commit()
    conn.close()
    return db


def test_prune_keeps_all_trade_data(tmp_path):
    db = _make_db(tmp_path)

    subprocess.run(["bash", str(SCRIPT), str(db)], check=True, capture_output=True,
                   env={**os.environ, "BACKUP_DIR": str(tmp_path / "bk")})

    conn = sqlite3.connect(db)
    count = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    assert count("trades") == 3000          # every row kept, skipped signals included
    assert count("rl_experience") == 2500
    assert count("candles") == 500          # newest 500 per symbol
    assert conn.execute("SELECT MIN(ts) FROM candles").fetchone()[0] == 200
    # last 2 days kept at full 30 s detail; older thinned to one per 5 minutes
    recent_cut = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    n_old = conn.execute("SELECT COUNT(*) FROM equity_curve WHERE ts < ?", (recent_cut,)).fetchone()[0]
    n_new = conn.execute("SELECT COUNT(*) FROM equity_curve WHERE ts >= ?", (recent_cut,)).fetchone()[0]
    assert n_old == 12 and n_new == 120
    conn.close()


def _run(db, **env):
    return subprocess.run(["bash", str(SCRIPT), str(db)], capture_output=True, text=True,
                          env={**os.environ, **{k: str(v) for k, v in env.items()}})


def test_backup_is_written_before_pruning_and_is_complete(tmp_path):
    db = _make_db(tmp_path)
    r = _run(db, BACKUP_DIR=tmp_path / "bk")
    assert r.returncode == 0
    backups = sorted((tmp_path / "bk").glob("trading-*.db.gz"))
    assert len(backups) == 1
    restored = tmp_path / "restored.db"
    restored.write_bytes(gzip.decompress(backups[0].read_bytes()))
    conn = sqlite3.connect(restored)
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 3000
    assert conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] == 700   # pre-prune snapshot
    conn.close()
    assert not list((tmp_path / "bk").glob(".trading-*"))                     # temp file cleaned up


def test_backup_rotation_keeps_newest(tmp_path):
    db = _make_db(tmp_path)
    bk = tmp_path / "bk"
    bk.mkdir()
    for i, age in enumerate((300, 200, 100)):                                  # three older backups
        f = bk / f"trading-2026010{i}T000000Z.db.gz"
        f.write_bytes(gzip.compress(b"x"))
        t = os.path.getmtime(f) - age
        os.utime(f, (t, t))
    assert _run(db, BACKUP_DIR=bk, BACKUP_KEEP=2).returncode == 0
    kept = sorted(p.name for p in bk.glob("trading-*.db.gz"))
    assert len(kept) == 2 and "trading-20260102T000000Z.db.gz" in kept         # newest old one + new


def test_failed_backup_skips_prune(tmp_path):
    db = _make_db(tmp_path)
    (tmp_path / "blocker").write_text("not a directory")
    r = _run(db, BACKUP_DIR=tmp_path / "blocker" / "bk")
    assert r.returncode != 0 and "BACKUP FAILED" in r.stdout
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] == 700   # nothing pruned
    conn.close()
