"""
rl_agent.py — Soft Actor-Critic (SAC) inference agent.

EDGE DEPLOYMENT PHILOSOPHY
───────────────────────────
Training SAC via backpropagation on a Raspberry Pi is thermally dangerous
and computationally prohibitive.  This module implements ONLY the actor
network's forward pass — a tiny 3-layer MLP in pure NumPy.

The offline_trainer.py on your laptop trains the full SAC (actor + two
critics + temperature α) using PyTorch, then exports the actor weights as
a .npz file.  This module loads those weights and runs inference in
microseconds with zero PyTorch overhead.

MDP Formulation
───────────────
State  S_t: [ml_prob, balance_ratio, unrealised_pnl_pct, drawdown_pct,
              atr_pct_norm, adx_norm, regime_enc, vol_norm]  → R^8

Action A_t: continuous ∈ [0, 1] — fraction of available capital to deploy.
            A=0: hold cash; A=1: maximum conviction.

Reward R_t: differential Sharpe ratio, penalised for max drawdown and fees.
            (computed in offline_trainer.py during training)

SAC objective (entropy-regularised):
  J(π) = Σ E[r(s,a) + α·H(π(·|s))]
  The entropy bonus prevents the policy collapsing into a deterministic
  bet-everything behaviour — crucial for financial RL stability.
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path

from config import SAC_STATE_DIM, SAC_HIDDEN_DIM, SAC_ACTION_DIM, SAC_WEIGHTS

log = logging.getLogger(__name__)


class SACActorNumpy:
    """
    Lightweight actor network: S → A.

    Architecture: Linear(8→32) → Tanh → Linear(32→32) → Tanh → Linear(32→1) → Sigmoid

    Sigmoid output maps the action to [0, 1] (position fraction).
    The offline trainer targets a squashed Gaussian; the deterministic
    mean (passed through Sigmoid) is used for live inference.

    Forward pass timing on Pi 5: ~40 µs — negligible vs other latencies.
    """

    def __init__(self) -> None:
        sd = SAC_STATE_DIM
        h  = SAC_HIDDEN_DIM
        ad = SAC_ACTION_DIM

        # Xavier uniform initialisation — sensible starting point before trained
        self.W1 = self._xavier(sd, h)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = self._xavier(h, h)
        self.b2 = np.zeros(h, dtype=np.float32)
        self.W3 = self._xavier(h, ad)
        self.b3 = np.zeros(ad, dtype=np.float32)

        self._trained = False
        self._load_weights()

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE  (called from asyncio via run_in_executor)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, state: np.ndarray) -> float:
        """
        Given state vector S_t, return position fraction ∈ [0, 1].

        If weights have not been loaded (no offline training yet), falls
        back to a conservative heuristic based on the ml_prob alone.
        """
        if not self._trained:
            # Fallback: scale position by ml_prob, cap at 0.5 before training
            ml_prob = float(state[0]) if len(state) > 0 else 0.5
            return max(0.0, min(0.5, (ml_prob - 0.5) * 1.5))

        h1  = np.tanh(state @ self.W1 + self.b1)
        h2  = np.tanh(h1 @ self.W2 + self.b2)
        raw = h2 @ self.W3 + self.b3
        # Sigmoid squash to [0, 1]
        action = 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
        return float(action[0])

    # ─────────────────────────────────────────────────────────────────────────
    # WEIGHT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def save_weights(self, path: Path = SAC_WEIGHTS) -> None:
        np.savez(str(path),
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 trained=np.array([True]))
        log.info("SAC actor weights saved → %s", path)

    def _load_weights(self, path: Path = SAC_WEIGHTS) -> bool:
        if not path.exists():
            log.info("No SAC weights found — using heuristic fallback")
            return False
        try:
            data = np.load(str(path))
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.W3 = data["W3"].astype(np.float32)
            self.b3 = data["b3"].astype(np.float32)
            self._trained = bool(data.get("trained", [False])[0])
            log.info("SAC actor weights loaded from %s (trained=%s)",
                     path, self._trained)
            return True
        except Exception as exc:
            log.error("Failed to load SAC weights: %s — using heuristic", exc)
            return False

    def reload_weights(self) -> bool:
        """Hot-reload weights after offline trainer completes."""
        return self._load_weights()

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier uniform initialisation."""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit,
                                 size=(fan_in, fan_out)).astype(np.float32)


# Module-level singleton (shared within a process, not across processes)
_actor: SACActorNumpy | None = None


def get_actor() -> SACActorNumpy:
    global _actor
    if _actor is None:
        _actor = SACActorNumpy()
    return _actor


def compute_position_fraction(state: np.ndarray) -> float:
    """
    Public entry point — compute position size fraction via SAC actor.
    Called from brain.py (via run_in_executor in the async bot).
    """
    return get_actor().forward(state)


def reload_sac_weights() -> None:
    """Called after offline trainer uploads new weights to the Pi."""
    actor = get_actor()
    actor.reload_weights()
    log.info("SAC actor weights hot-reloaded")
