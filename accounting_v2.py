"""
accounting_v2.py — True net PnL after fees (V2).

Theory
------
Gross PnL is mark-to-market price delta × size. Economic PnL must subtract:
  • Exchange taker fees (conservative assumption — many retail orders match as taker)
  • Slippage already partially modeled in exec_price; we add explicit fee schedule
  • Optional funding accrual for perpetual-style carry (spot margin uses borrow interest;
    we approximate with a small hourly carry rate when enabled)

Kelly / RL agents trained on gross PnL overestimate edge; net PnL aligns rewards with
withdrawable economics.
"""

from __future__ import annotations

from config import TAKER_FEE_BPS, FEE_GATE_ROUND_TRIP


def taker_fee_rate() -> float:
    return TAKER_FEE_BPS / 10_000.0


def round_trip_fee_rate() -> float:
    """Open + close both pay taker fee (worst case for sizing safety)."""
    return 2.0 * taker_fee_rate()


def slippage_cost_rate() -> float:
    """V2.2: disable fee/friction offsets for pure alpha tests."""
    return 0.0


def break_even_price_offset_fraction() -> float:
    """
    Fractional distance beyond flat PnL where the trade is truly scratch after fees.

    Long : BE price = entry × (1 + δ)
    Short: BE price = entry × (1 − δ)
    """
    return 0.0


def entry_exit_fees_notional(entry_notional: float, exit_notional: float) -> float:
    """
    Round-trip exchange fee using FEE_GATE_ROUND_TRIP (0.12% total = 0.06% × 2 legs).
    Uses the average of entry and exit notional to handle asymmetric legs correctly.

    V2.3: fees re-enabled so that net_pnl, fee_total, and SAC rewards reflect
    real economics.  The fee gate in bot.py already used 0.12% as its block
    threshold — this makes the accounting layer consistent with it.
    """
    avg_notional = (float(entry_notional) + float(exit_notional)) / 2.0
    return max(0.0, avg_notional * FEE_GATE_ROUND_TRIP)


def net_realized_pnl(
    gross_pnl: float,
    entry_notional: float,
    exit_notional: float,
    hold_hours: float = 0.0,
) -> tuple[float, float]:
    """
    Returns (fee_total, net_pnl).
    V2.3: deducts 0.12% round-trip taker fee from gross PnL so that
    realised_pnl_net in the portfolio table tracks true withdrawable economics.
    """
    fees = entry_exit_fees_notional(entry_notional, exit_notional)
    net = gross_pnl - fees
    return fees, net


def position_equity_components(
    *,
    side: str,
    avg_cost: float,
    quantity: float,
    mark_price: float,
    margin_reserved: float = 0.0,
) -> tuple[float, float, float]:
    """
    Strict position accounting components.

    Returns (position_value, unrealised_pnl, equity_contribution) where:
      equity_contribution = position_value + unrealised_pnl

    Long:
      position_value   = avg_cost * qty
      unrealised_pnl   = (mark - avg_cost) * qty
      contribution     = mark * qty

    Short:
      position_value   = margin_reserved
      unrealised_pnl   = (avg_cost - mark) * qty
      contribution     = margin_reserved + unrealised_pnl
    """
    s = str(side or "long").lower()
    avg = float(avg_cost or 0.0)
    qty = float(quantity or 0.0)
    mark = float(mark_price or 0.0)
    margin = float(margin_reserved or 0.0)

    if s == "short":
        pos_value = margin
        unr = (avg - mark) * qty
    else:
        pos_value = avg * qty
        unr = (mark - avg) * qty
    return pos_value, unr, pos_value + unr


def shaped_reward_net(
    gross_pnl: float,
    hold_candles: int,
    trade_value: float,
    hold_hours: float,
    *,
    candle_cost: float,
) -> float:
    """RL reward on net economics + per-candle opportunity cost."""
    _, net = net_realized_pnl(gross_pnl, trade_value, trade_value, hold_hours)
    return net - hold_candles * candle_cost
