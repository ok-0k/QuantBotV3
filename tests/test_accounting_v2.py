"""Characterization tests for accounting_v2 — the fee/PnL money math.

These pin CURRENT behaviour exactly (FEE_GATE_ROUND_TRIP = 0.0020).
Any intentional change to fee handling must update these numbers in the
same commit, with an explanation.

Wave 3 fix: FEE_GATE_ROUND_TRIP's default changed from 0.0012 to 0.0020.
The old value silently assumed a BNB/VIP fee discount (0.06%/leg) with no
code checking the account actually had it enabled; 0.0020 is the honest,
undiscounted standard Binance spot taker rate (0.10%/leg x 2). The discount
is now opt-in via config.USE_BNB_FEE_DISCOUNT, defaulting to False.
"""

import pytest

from accounting_v2 import (
    entry_exit_fees_notional,
    net_realized_pnl,
    position_equity_components,
    round_trip_fee_rate,
    shaped_reward_net,
    taker_fee_rate,
)
from config import FEE_GATE_ROUND_TRIP


def test_fee_constants_are_legacy_zero_plus_active_gate():
    # Legacy V2.2 constants are zero; all live accounting uses FEE_GATE_ROUND_TRIP.
    assert taker_fee_rate() == 0.0
    assert round_trip_fee_rate() == 0.0
    assert FEE_GATE_ROUND_TRIP == 0.0020


def test_round_trip_fee_symmetric_notional():
    # $1000 in / $1000 out -> 1000 * 0.0020 = $2.00
    assert entry_exit_fees_notional(1000.0, 1000.0) == pytest.approx(2.00)


def test_round_trip_fee_asymmetric_legs_use_average():
    # avg(1000, 500) = 750 -> 750 * 0.0020 = $1.50
    assert entry_exit_fees_notional(1000.0, 500.0) == pytest.approx(1.50)


def test_round_trip_fee_never_negative():
    assert entry_exit_fees_notional(-100.0, -100.0) == 0.0


def test_net_realized_pnl_deducts_fees():
    fee, net = net_realized_pnl(10.0, 1000.0, 1000.0)
    assert fee == pytest.approx(2.00)
    assert net == pytest.approx(8.00)


def test_net_realized_pnl_loss_gets_worse_after_fees():
    fee, net = net_realized_pnl(-5.0, 1000.0, 1000.0)
    assert fee == pytest.approx(2.00)
    assert net == pytest.approx(-7.00)


def test_long_equity_components():
    pos_value, unreal, contrib = position_equity_components(
        side="long", avg_cost=100.0, quantity=2.0, mark_price=110.0,
    )
    assert pos_value == pytest.approx(200.0)   # cost basis
    assert unreal == pytest.approx(20.0)       # (110-100) * 2
    assert contrib == pytest.approx(220.0)     # mark * qty


def test_short_equity_components():
    pos_value, unreal, contrib = position_equity_components(
        side="short", avg_cost=100.0, quantity=2.0, mark_price=90.0,
        margin_reserved=40.0,
    )
    assert pos_value == pytest.approx(40.0)    # collateral
    assert unreal == pytest.approx(20.0)       # (100-90) * 2 profit
    assert contrib == pytest.approx(60.0)


def test_short_equity_components_squeeze_loss():
    _, unreal, contrib = position_equity_components(
        side="short", avg_cost=100.0, quantity=2.0, mark_price=115.0,
        margin_reserved=40.0,
    )
    assert unreal == pytest.approx(-30.0)
    assert contrib == pytest.approx(10.0)


def test_degenerate_inputs_return_zeroes():
    assert position_equity_components(
        side="long", avg_cost=None, quantity=None, mark_price=None,
    ) == (0.0, 0.0, 0.0)


def test_shaped_reward_net_subtracts_fees_and_time_cost():
    # net = 10 - 2.0 = 8.0; time cost = 100 candles * 0.00005 = 0.005
    r = shaped_reward_net(10.0, 100, 1000.0, 1.6667, candle_cost=0.00005)
    assert r == pytest.approx(7.995)
