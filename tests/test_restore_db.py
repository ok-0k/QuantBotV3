"""
scripts/restore_db.sh refuses while services run, verifies the backup, keeps the
old files aside; prune_db.sh keeps a snapshot that fails quick_check and prunes
nothing.
"""

import gzip
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESTORE = ROOT / "scripts" / "restore_db.sh"
PRUNE = ROOT / "prune_db.sh"
pytestmark = pytest.mark.skipif(shutil.which("sqlite3") is None, reason="sqlite3 CLI not installed")


def _db(path, n_trades):
    conn = sqlite3.connect(path)
    conn.executescript("""CREATE TABLE trades (id INTEGER PRIMARY KEY, ts TEXT);
                          CREATE TABLE candles (symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL,
                                                close REAL, volume REAL, PRIMARY KEY (symbol, ts));
                          CREATE TABLE equity_curve (ts TEXT, equity REAL);
                          CREATE TABLE rl_experience (id INTEGER PRIMARY KEY, ts TEXT);""")
    conn.executemany("INSERT INTO trades (ts) VALUES (?)", [(str(i),) for i in range(n_trades)])
    conn.executemany("INSERT INTO candles VALUES ('A',?,1,1,1,1,1)", [(i,) for i in range(600)])
    conn.commit()
    conn.close()


def _fake_systemctl(tmp_path, active):
    f = tmp_path / "systemctl"
    f.write_text(f"#!/bin/bash\nexit {0 if active else 3}\n")
    f.chmod(0o755)
    return str(f)


@pytest.fixture()
def live(tmp_path):
    db = tmp_path / "trading.db"
    _db(db, 5)
    (tmp_path / "trading.db-wal").write_bytes(b"stale wal")
    src = tmp_path / "src.db"
    _db(src, 42)
    backup = tmp_path / "trading-x.db.gz"
    backup.write_bytes(gzip.compress(src.read_bytes()))
    return db, backup


def _restore(tmp_path, db, backup, *extra, active=False):
    env = {**os.environ, "DB": str(db), "SYSTEMCTL": _fake_systemctl(tmp_path, active)}
    return subprocess.run(["bash", str(RESTORE), str(backup), *extra], capture_output=True, text=True, env=env)


def _count(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    finally:
        conn.close()


def test_refuses_while_a_service_runs(tmp_path, live):
    db, backup = live
    r = _restore(tmp_path, db, backup, "--yes", active=True)
    assert r.returncode == 1 and "Refusing" in r.stdout
    assert _count(db) == 5


def test_dry_run_verifies_without_changing_anything(tmp_path, live):
    db, backup = live
    r = _restore(tmp_path, db, backup)
    assert r.returncode == 0 and "42 trade rows" in r.stdout
    assert (tmp_path / "trading.db-wal").exists()                      # check before opening the DB
    assert _count(db) == 5


def test_restore_replaces_db_and_keeps_old_files_aside(tmp_path, live):
    db, backup = live
    r = _restore(tmp_path, db, backup, "--yes")
    assert r.returncode == 0
    assert _count(db) == 42
    assert not (tmp_path / "trading.db-wal").exists()                 # stale WAL moved away
    aside = next(tmp_path.glob("pre-restore-*"))
    assert (aside / "trading.db").exists() and (aside / "trading.db-wal").exists()


def test_corrupt_backup_is_rejected(tmp_path, live):
    db, backup = live
    backup.write_bytes(gzip.compress(b"not a database" * 100))
    r = _restore(tmp_path, db, backup, "--yes")
    assert r.returncode == 1 and "integrity_check" in r.stdout
    assert not list(tmp_path.glob(".restore-*"))                       # temp file cleaned up
    assert _count(db) == 5


def test_prune_keeps_snapshot_that_fails_quick_check(tmp_path):
    db = tmp_path / "t.db"
    _db(db, 10)
    real = shutil.which("sqlite3")
    wrapper = tmp_path / "bin" / "sqlite3"
    wrapper.parent.mkdir()
    wrapper.write_text(f'#!/bin/bash\nif [[ "$*" == *quick_check* ]]; then echo "*** page 3 corrupt"; exit 0; fi\nexec {real} "$@"\n')
    wrapper.chmod(0o755)
    env = {**os.environ, "BACKUP_DIR": str(tmp_path / "bk"), "PATH": f"{wrapper.parent}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(PRUNE), str(db)], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "BACKUP CHECK FAILED" in r.stdout
    assert list((tmp_path / "bk").glob("*.CHECK-FAILED.db"))
    assert not list((tmp_path / "bk").glob("trading-*.db.gz"))
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0] == 600   # nothing pruned
    conn.close()
