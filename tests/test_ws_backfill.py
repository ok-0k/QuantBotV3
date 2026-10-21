"""
Websocket reconnect backfill: every connect but the process's first (after
an error or a clean server close) REST-backfills the candle history before
the stream is consumed, however short the drop.
A keepalive timeout is raised 10-30 s after the last message, so the gap
measured from the error never showed how many closed candles were lost
(2026-10-19 16:38: a ~1 s reconnect missed the 16:38 close on ~46 symbols).
"""

import asyncio
import types

import pytest

import bot


class _FakeWS:
    """Async-iterable stand-in for a websockets connection. `script` items
    are messages to yield, exceptions to raise, or callables to run."""

    def __init__(self, script):
        self.script = script

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                item()
                continue
            yield item


@pytest.fixture()
def listener(monkeypatch):
    """Run bot.websocket_listener against scripted connections; returns the
    ordered event log (connect / backfill / msg / wait:<secs>). Reconnect
    waits and the uptime clock are fake: nothing really sleeps, and
    listener.advance(secs) moves the clock (e.g. inside a script)."""
    events = []
    clock = [1000.0]
    monkeypatch.setattr(bot, "_shutdown_event", asyncio.Event())
    monkeypatch.setattr(bot, "_ws_down_since", None)

    async def fake_backfill():
        events.append("backfill")
        return {}

    async def fake_handle(raw_msg, loop):
        events.append(f"msg:{raw_msg}")

    async def fake_wait(secs):
        events.append(f"wait:{secs:g}")
        clock[0] += secs
        await asyncio.sleep(0)
        return False                      # no shutdown during the wait

    def advance(secs):
        clock[0] += secs

    monkeypatch.setattr(bot, "_load_history_for_all_symbols", fake_backfill)
    monkeypatch.setattr(bot, "_handle_ws_message", fake_handle)
    monkeypatch.setattr(bot, "_ws_reconnect_wait", fake_wait)
    monkeypatch.setattr(bot, "_ws_clock", lambda: clock[0])

    def run(*scripts):
        conns = iter(scripts)

        def connect(url, **kw):
            events.append("connect")
            return _FakeWS(next(conns))

        monkeypatch.setattr(bot, "websockets", types.SimpleNamespace(connect=connect))

        async def go():
            await asyncio.wait_for(bot.websocket_listener(asyncio.get_running_loop()), timeout=10)

        asyncio.run(go())
        return events

    run.advance = advance
    return run


def test_short_drop_backfills_before_consuming_the_stream(listener):
    stop = bot._shutdown_event.set
    events = listener(
        ["k1", Exception("sent 1011 (internal error) keepalive ping timeout; no close frame received")],
        ["k2", stop, "k3"],
    )
    # reconnected ~1 s after the error, far below the old 90 s threshold
    assert events == ["connect", "msg:k1", "wait:1", "connect", "backfill", "msg:k2"]
    assert bot._ws_down_since is None


def test_first_connect_does_not_backfill(listener):
    events = listener(["k1", bot._shutdown_event.set, "k2"])
    assert events == ["connect", "msg:k1"]
    assert "backfill" not in events


def test_clean_server_close_backfills_on_the_next_connect(listener):
    # Binance closes connections itself: the `async for` just ends, no exception
    events = listener(["k1"], ["k2", bot._shutdown_event.set, "k3"])
    assert events == ["connect", "msg:k1", "wait:1", "connect", "backfill", "msg:k2"]
    assert bot._ws_down_since is None


def test_immediate_clean_closes_back_off_and_cap_backfills(listener):
    # a server that keeps closing at once: waits grow to the 60 s cap, and
    # every backfill comes after its own wait (no request storm)
    events = listener(*[[] for _ in range(9)], [bot._shutdown_event.set, "k"])
    waits = [e for e in events if e.startswith("wait:")]
    assert waits == ["wait:1", "wait:2", "wait:4", "wait:8", "wait:16", "wait:32",
                     "wait:60", "wait:60", "wait:60"]
    assert events.count("backfill") == len(waits)
    kinds = [e for e in events if e == "backfill" or e.startswith("wait:")]
    assert kinds == [k for w in waits for k in (w, "backfill")]   # strictly alternating


@pytest.mark.parametrize("uptime, wait_after", [(60, "wait:1"), (59, "wait:4")])
def test_backoff_resets_only_after_a_connection_stayed_up_60s(listener, uptime, wait_after):
    drop = Exception("keepalive ping timeout")
    events = listener(
        [],                                                  # clean close at once: wait 1
        [drop],                                              # error at once: wait 2
        ["k1", lambda: listener.advance(uptime), drop],      # up `uptime` s, then drops
        [bot._shutdown_event.set, "k2"],
    )
    assert [e for e in events if e.startswith("wait:")] == ["wait:1", "wait:2", wait_after]
