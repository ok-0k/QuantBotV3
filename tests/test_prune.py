"""
prune_db.sh must never delete trade data (trades / rl_experience are the
learning record and lifetime history) while still trimming candles and the
equity snapshot series.
"""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "prune_db.sh"

pytestmark = pytest.mark.skipif(shutil.which("sqlite3") is None, reason="sqlite3 CLI not installed")


def test_prune_keeps_all_trade_data(tmp_path):
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
    conn.executemany("INSERT INTO equity_curve VALUES (?,1)", [(str(i),) for i in range(6000)])
    conn.commit()
    conn.close()

    subprocess.run(["bash", str(SCRIPT), str(db)], check=True, capture_output=True)

    conn = sqlite3.connect(db)
    count = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    assert count("trades") == 3000          # every row kept, skipped signals included
    assert count("rl_experience") == 2500
    assert count("candles") == 500          # newest 500 per symbol
    assert conn.execute("SELECT MIN(ts) FROM candles").fetchone()[0] == 200
    assert count("equity_curve") == 5000
    conn.close()
