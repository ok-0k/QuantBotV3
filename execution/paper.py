"""Paper adapter — simulated fills, identical to the historical inline maths."""

from __future__ import annotations

from execution.base import ExecutionAdapter, Fill, OrderIntent


class PaperAdapter(ExecutionAdapter):
    """
    Fills instantly at intent.limit_price (the slippage-adjusted exec_price
    bot.py has always computed) for the full requested quantity. No I/O, no
    failure modes — behaviour is byte-for-byte what _execute_trade did inline,
    which the executor characterization tests pin.
    """

    name = "paper"

    def execute(self, intent: OrderIntent) -> Fill:
        if intent.qty <= 0 or intent.limit_price <= 0:
            return Fill(status="REJECTED", note="degenerate qty/price")
        return Fill(
            status="FILLED",
            qty=float(intent.qty),
            price=float(intent.limit_price),
            note="paper fill",
        )

    def reconcile(self, client_order_id: str) -> Fill | None:
        # Paper fills are synchronous with the DB write; a non-terminal journal
        # row after a restart means the process died mid-commit — nothing was
        # booked, so the order is safely treated as never-executed.
        return Fill(status="REJECTED", qty=0.0, price=0.0, note="paper reconcile: not executed")
