"""Execution adapter interface, order/fill datatypes, deterministic order ids."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OrderIntent:
    """Everything the executor knows at the moment it commits to a fill."""
    client_order_id: str
    symbol: str
    action: str          # buy | sell | short | cover
    side: str            # long | short
    qty: float
    ref_price: float     # raw signal price (pre-slippage)
    limit_price: float   # slippage-adjusted price paper mode fills at
    candle_ts: Optional[int] = None


@dataclass(frozen=True)
class Fill:
    """
    Outcome of an execution attempt.

    status: FILLED | PARTIALLY_FILLED | REJECTED | FAILED
      FILLED           — full qty executed; book qty @ price.
      PARTIALLY_FILLED — qty (< requested) executed; book the actual fill.
      REJECTED         — exchange/validator said no; nothing executed.
      FAILED           — network/infra exhausted; nothing known-executed
                         (journal row stays for reconciliation).
    """
    status: str
    qty: float = 0.0
    price: float = 0.0
    exchange_order_id: Optional[str] = None
    note: Optional[str] = None

    @property
    def executed(self) -> bool:
        return self.status in ("FILLED", "PARTIALLY_FILLED") and self.qty > 0


def make_client_order_id(symbol: str, action: str, candle_ts: Optional[int]) -> str:
    """
    Deterministic id per (symbol, action, candle). Duplicate signals from the
    same candle — including retries after a timeout — map to the SAME id, so
    the orders journal and the exchange (newClientOrderId) both dedupe them.
    Binance allows <= 36 chars from [a-zA-Z0-9-_]; 'qb-' + 24 hex = 27 chars.
    """
    raw = f"{symbol}|{action}|{candle_ts if candle_ts is not None else 'na'}"
    return "qb-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


class ExecutionAdapter(ABC):
    """One method matters: turn an intent into a Fill. Implementations must
    never raise for ordinary trading outcomes — encode them in Fill.status."""

    name: str = "abstract"

    @abstractmethod
    def execute(self, intent: OrderIntent) -> Fill:
        ...

    def reconcile(self, client_order_id: str) -> Optional[Fill]:
        """Best-effort status lookup for startup reconciliation. None = unknown."""
        return None
