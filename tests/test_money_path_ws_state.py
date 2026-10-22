"""
Money path, inputs and memory: what a websocket kline does before any trade
decision (bot._handle_ws_message: candle cache, closed-candle persistence, the
ML probability that gates entries, scheduling of the evaluation), and the
adaptive edge profiles whose size_mult scales every entry
(bot._persist_adaptive_edge_profiles / _restore_adaptive_edge_profiles).
"""

import asyncio
import json

import pytest

import bot
import db
from config import CANDLE_LIMIT, EDGE_SIZE_MAX_MULT, EDGE_SIZE_MIN_MULT, SYMBOLS
from conftest import make_candles

SYM = SYMBOLS[3]


def _kline(t, close, closed=False, sym=SYM, **over):
    k = {"t": t, "o": close, "h": close + 0.1, "l": close - 0.1, "c": close, "v": 5.0, "x": closed}
    k.update(over)
    return json.dumps({"data": {"s": sym, "k": k}})


@pytest.fixture()
def ws(clean_db, monkeypatch):
    """Runs messages through the real handler; records the tasks it schedules."""
    rec = {"tick": [], "eval": [], "train": [], "predict": []}

    async def tick(sym, candle, loop):
        rec["tick"].append((sym, candle["close"]))

    async def evaluate(sym, candles, ml_prob, loop):
        rec["eval"].append((sym, len(candles), ml_prob))

    async def train(loop, sym, candles):
        rec["train"].append(sym)

    def predict(candles):
        rec["predict"].append(len(candles))
        return 0.07
    monkeypatch.setattr(bot, "_tick_exit_check", tick)
    monkeypatch.setattr(bot, "_evaluate_and_trade", evaluate)
    monkeypatch.setattr(bot, "_run_incremental_train", train)
    monkeypatch.setattr(bot, "predict_signal", predict)
    monkeypatch.setattr(bot, "_candles_cache", {})
    monkeypatch.setattr(bot, "_closed_candle_counts", {})
    monkeypatch.setattr(bot, "_ws_heartbeat_last", 0.0)
    bot.brain.ml_probs.clear()

    def send(*msgs):
        async def go():
            loop = asyncio.get_running_loop()
            for m in msgs:
                await bot._handle_ws_message(m, loop)
            await asyncio.sleep(0.02)                    # let scheduled tasks run
        asyncio.run(go())
        return rec
    return send


def _seed_cache(n):
    candles = make_candles(n=n, start_price=100.0)
    bot._candles_cache[SYM] = candles
    return candles


@pytest.mark.parametrize("msg", [
    "{not json",
    json.dumps({"data": {"s": "NOPEUSDT", "k": {"t": 1, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}}}),
    json.dumps({"data": {"s": SYM}}),
])
def test_junk_messages_are_ignored(ws, msg):
    rec = ws(msg)
    assert bot._candles_cache == {} and rec["tick"] == []


def test_malformed_kline_is_logged_and_dropped(ws, caplog):
    with caplog.at_level("WARNING"):
        rec = ws(_kline(60_000, 1.0, c="n/a"))
    assert bot._candles_cache == {} and rec["tick"] == []
    assert any(f"Malformed kline for {SYM}" in r.getMessage() for r in caplog.records)


def test_closed_candle_is_persisted_but_not_traded_before_sixty_bars(ws):
    rec = ws(_kline(60_000, 101.5, closed=True))
    assert db.get_candles(SYM)[-1]["close"] == 101.5
    assert float(db.get_portfolio_stat(f"last_price_{SYM}")) == 101.5
    assert bot._closed_candle_counts[SYM] == 1
    assert rec["tick"] == [(SYM, 101.5)]
    assert rec["predict"] == [] and rec["eval"] == []


def test_closed_candle_refreshes_the_ml_gate_and_schedules_an_evaluation(ws):
    candles = _seed_cache(80)
    rec = ws(_kline(candles[-1]["time"] + 60_000, 99.0, closed=True))
    assert rec["predict"] == [81]
    assert bot.brain.ml_probs[SYM] == 0.07
    assert db.get_ml_prob(SYM) == 0.07
    assert rec["eval"] == [(SYM, 81, 0.07)]
    assert rec["train"] == []


def test_ml_failure_falls_back_to_a_neutral_probability(ws, monkeypatch, caplog):
    candles = _seed_cache(80)

    def boom(c):
        raise ValueError("model file truncated")
    monkeypatch.setattr(bot, "predict_signal", boom)
    with caplog.at_level("WARNING"):
        rec = ws(_kline(candles[-1]["time"] + 60_000, 99.0, closed=True))
    assert rec["eval"] == [(SYM, 81, 0.5)]               # 0.5 never clears the 0.80 entry gate
    assert bot.brain.ml_probs[SYM] == 0.5
    assert any(f"ML predict error for {SYM}" in r.getMessage() for r in caplog.records)


def test_incremental_training_is_scheduled_on_its_cadence(ws, monkeypatch):
    candles = _seed_cache(80)
    monkeypatch.setattr(bot, "ML_UPDATE_EVERY", 2)
    t = candles[-1]["time"]
    rec = ws(_kline(t + 60_000, 99.0, closed=True), _kline(t + 120_000, 98.0, closed=True))
    assert rec["train"] == [SYM]                         # on the 2nd closed candle only
    assert len(rec["eval"]) == 2


def test_cache_is_capped_at_the_candle_limit(ws):
    times = [c["time"] for c in _seed_cache(CANDLE_LIMIT)]
    ws(_kline(times[-1] + 60_000, 99.0))
    cache = bot._candles_cache[SYM]
    assert len(cache) == CANDLE_LIMIT
    assert cache[0]["time"] == times[1] and cache[-1]["close"] == 99.0      # oldest bar dropped


def test_heartbeat_write_failure_does_not_stop_the_tick(ws, monkeypatch):
    def broken(k, v):
        if k == "ws_last_msg":
            raise RuntimeError("database is locked")
    monkeypatch.setattr(bot, "set_portfolio_stat", broken)
    rec = ws(_kline(60_000, 100.0))
    assert rec["tick"] == [(SYM, 100.0)]
    assert bot._candles_cache[SYM][-1]["close"] == 100.0


# ── adaptive edge profiles (they scale every entry's size) ──────────────────

@pytest.fixture()
def profiles(clean_db):
    bot.brain._edge_profiles.clear()
    yield bot.brain._edge_profiles
    bot.brain._edge_profiles.clear()


def test_edge_profiles_survive_a_restart(profiles):
    bot.brain._edge_profile("short", "ranging")["size_mult"] = 0.4
    bot.brain._edge_profile("short", "trend_down")["rr_mult"] = 1.2
    bot._persist_adaptive_edge_profiles()
    profiles.clear()
    assert bot._restore_adaptive_edge_profiles() == 2
    assert profiles["short:ranging"]["size_mult"] == 0.4
    assert profiles["short:trend_down"]["rr_mult"] == 1.2


def test_restored_size_multipliers_are_clamped(profiles):
    db.save_brain_key(bot._EDGE_PROFILE_DB_KEY, {
        "short:ranging": {"size_mult": 50.0, "rr_mult": 1.0},
        "short:trend_down": {"size_mult": -3.0},
        "junk": "not a dict"})
    assert bot._restore_adaptive_edge_profiles() == 2
    assert profiles["short:ranging"]["size_mult"] == EDGE_SIZE_MAX_MULT
    assert profiles["short:trend_down"]["size_mult"] == EDGE_SIZE_MIN_MULT
    assert "junk" not in profiles


@pytest.mark.parametrize("setup, text", [
    (lambda mp: None, "No edge profiles found"),
    (lambda mp: db.save_brain_key(bot._EDGE_PROFILE_DB_KEY, [1, 2]), "is not a dict"),
    (lambda mp: mp.setattr(bot, "load_brain_key", lambda *a: 1 / 0), "Edge profile DB read failed"),
    (lambda mp: (db.save_brain_key(bot._EDGE_PROFILE_DB_KEY, {"short:ranging": {"size_mult": 0.5}}),
                 mp.setattr(bot.brain, "import_edge_profiles", lambda p: 1 / 0)), "import_edge_profiles failed"),
])
def test_restore_starts_clean_on_any_bad_state(profiles, monkeypatch, caplog, setup, text):
    setup(monkeypatch)
    with caplog.at_level("INFO"):
        assert bot._restore_adaptive_edge_profiles() == 0
    assert profiles == {}
    assert any(text in r.getMessage() for r in caplog.records)


def test_persist_failure_is_logged_not_raised(profiles, monkeypatch, caplog):
    monkeypatch.setattr(bot.brain, "export_edge_profiles", lambda: {"k": {1, 2}})   # sets: not JSON
    with caplog.at_level("ERROR"):
        bot._persist_adaptive_edge_profiles()
    assert db.load_brain_key(bot._EDGE_PROFILE_DB_KEY, None) is None
    assert any("_persist_adaptive_edge_profiles FAILED" in r.getMessage() for r in caplog.records)
