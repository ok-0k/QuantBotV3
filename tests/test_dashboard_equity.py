"""Equity chart: long windows read their time range; thinning is time-uniform."""

from datetime import datetime, timedelta, timezone

import dashboard


def _pt(t):
    return {"time": t.strftime("%Y-%m-%d %H:%M:%S"), "equity": 1.0}


def test_downsample_is_time_uniform_on_mixed_density():
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    sparse = [_pt(now - timedelta(days=7) + timedelta(minutes=5 * i)) for i in range(5 * 288)]   # days 7..2, 5 min
    dense = [_pt(now - timedelta(days=2) + timedelta(seconds=30 * i)) for i in range(2 * 2880)]  # last 2 days, 30 s
    out = dashboard._downsample_equity(sparse + dense, 350)
    assert len(out) <= 350 and out[0] == sparse[0] and out[-1] == dense[-1]
    # the last 2 days are 2/7 of the time span, so they should get ~2/7 of the points
    cut = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    recent_share = sum(p["time"] >= cut for p in out) / len(out)
    assert 0.24 < recent_share < 0.34


def test_downsample_passthrough_and_degenerate():
    pts = [_pt(datetime(2026, 1, 1, tzinfo=timezone.utc))] * 3
    assert dashboard._downsample_equity(pts, 10) == pts
    assert dashboard._downsample_equity(pts * 10, 5) == [pts[0], pts[0]]   # zero time span


def test_equity_curve_since_reads_time_window(clean_db):
    now = datetime.now(timezone.utc)
    with clean_db.get_db() as conn:
        for d in (40, 20, 3, 0):
            conn.execute("INSERT INTO equity_curve (ts, equity) VALUES (?, ?)",
                         ((now - timedelta(days=d)).isoformat(), float(d)))
    got = clean_db.get_equity_curve_since((now - timedelta(days=30)).isoformat())
    assert [p["equity"] for p in got] == [20.0, 3.0, 0.0]          # oldest first, 40d excluded
    assert len(got[0]["time"]) == 19 and "T" not in got[0]["time"]
