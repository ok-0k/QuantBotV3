"""
execution — pluggable order-execution backends.

    paper   : simulated fills at the slippage-adjusted price (default;
              byte-for-byte the maths bot.py has always used)
    testnet : Binance Spot testnet (testnet.binance.vision) — real order
              lifecycle with worthless funds

There is intentionally NO live-mainnet backend in this package.
"""

from __future__ import annotations

from config import (
    BINANCE_TESTNET_API_KEY,
    BINANCE_TESTNET_API_SECRET,
    EXECUTION_MODE,
)
from execution.base import ExecutionAdapter, Fill, OrderIntent, make_client_order_id
from execution.paper import PaperAdapter

_adapter: ExecutionAdapter | None = None


def get_execution_adapter() -> ExecutionAdapter:
    """Process-wide adapter singleton, selected by EXECUTION_MODE."""
    global _adapter
    if _adapter is not None:
        return _adapter

    mode = EXECUTION_MODE
    if mode == "paper":
        _adapter = PaperAdapter()
    elif mode == "testnet":
        if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_API_SECRET:
            raise RuntimeError(
                "EXECUTION_MODE=testnet requires BINANCE_TESTNET_API_KEY and "
                "BINANCE_TESTNET_API_SECRET in the environment "
                "(~/.config/quant-bot/env). Refusing to start."
            )
        from execution.binance_testnet import BinanceTestnetAdapter
        _adapter = BinanceTestnetAdapter()
    else:
        raise RuntimeError(
            f"Unknown EXECUTION_MODE={mode!r} — expected 'paper' or 'testnet'."
        )
    return _adapter


__all__ = [
    "ExecutionAdapter",
    "Fill",
    "OrderIntent",
    "PaperAdapter",
    "get_execution_adapter",
    "make_client_order_id",
]
