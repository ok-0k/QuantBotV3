"""
rl_agent.py — Soft Actor-Critic (SAC) inference agent.
V2 Architecture: 13-input → LayerNorm+Tanh → LayerNorm+Tanh → mu head → tanh squash
Keys: W1, b1, W2, b2, W_mu, b_mu, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta
"""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from config import SAC_STATE_DIM, SAC_WEIGHTS

log = logging.getLogger(__name__)


class SACActorNumpy:
    """
    Lightweight SAC actor network: S → A ∈ (-1, 1).

    Architecture:
        Linear(13→32) → LayerNorm → Tanh
        → Linear(32→32) → LayerNorm → Tanh
        → Linear(32→1)  [mu head]
        → tanh squash
    """

    def __init__(self) -> None:
        self._trained = False
        self._load_weights()

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _layer_norm(
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Apply Layer Normalization:
            y = gamma * (x - mu) / (sigma + eps) + beta
        Operates over the last axis so it works for both 1-D and 2-D inputs.
        """
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True)
        return gamma * (x - mu) / (sigma + eps) + beta

    @staticmethod
    def _tanh_squash(x: np.ndarray) -> np.ndarray:
        """Squash raw output to the open interval (-1, 1) via tanh."""
        return np.tanh(np.clip(x, -10.0, 10.0))

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE  (called from asyncio via run_in_executor)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, state: np.ndarray) -> float:
        """
        Given a 13-point state feature vector, return a position-size signal
        in the open interval (-1, 1).

        Falls back to a simple heuristic when no weights are loaded.
        """
        if not self._trained:
            # Heuristic fallback: map state[0] ∈ [0,1] to signal ∈ [-0.5, 0.5]
            prob = float(state[0]) if len(state) > 0 else 0.5
            return float(np.clip((prob - 0.5) * 1.5, -0.5, 0.5))

        s = state.astype(np.float32)

        # Layer 1: linear → layer-norm → tanh
        h1 = np.tanh(
            self._layer_norm(s @ self.W1 + self.b1, self.ln1_gamma, self.ln1_beta)
        )

        # Layer 2: linear → layer-norm → tanh
        h2 = np.tanh(
            self._layer_norm(h1 @ self.W2 + self.b2, self.ln2_gamma, self.ln2_beta)
        )

        # Output (mu) head — no activation before squash
        raw = h2 @ self.W_mu + self.b_mu

        # Tanh squash → action ∈ (-1, 1)
        action = self._tanh_squash(raw)

        # Return a plain Python float; squeeze in case output dim > 1
        return float(np.squeeze(action))

    # ─────────────────────────────────────────────────────────────────────────
    # WEIGHT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _load_weights(self, path: Path = SAC_WEIGHTS) -> bool:
        """
        Load V2 weights from *path*.

        Expected keys inside the .npz archive:
            W1, b1, W2, b2, W_mu, b_mu,
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta
        """
        if not path.exists():
            log.info(
                "No SAC weights found at %s — using heuristic fallback",
                path.resolve(),
            )
            return False

        try:
            data = np.load(str(path))

            # Hidden layer 1
            self.W1 = data["W1"].astype(np.float32)   # (13, 32)
            self.b1 = data["b1"].astype(np.float32)   # (32,)

            # Hidden layer 2
            self.W2 = data["W2"].astype(np.float32)   # (32, 32)
            self.b2 = data["b2"].astype(np.float32)   # (32,)

            # Output (mu) head
            self.W_mu = data["W_mu"].astype(np.float32)  # (32, 1)
            self.b_mu = data["b_mu"].astype(np.float32)  # (1,)

            # LayerNorm parameters — layer 1
            self.ln1_gamma = data["ln1_gamma"].astype(np.float32)  # (32,)
            self.ln1_beta  = data["ln1_beta"].astype(np.float32)   # (32,)

            # LayerNorm parameters — layer 2
            self.ln2_gamma = data["ln2_gamma"].astype(np.float32)  # (32,)
            self.ln2_beta  = data["ln2_beta"].astype(np.float32)   # (32,)

            self._trained = True
            log.info("SAC V2 actor weights loaded successfully from %s", path)
            return True

        except KeyError as exc:
            log.error(
                "Missing key in SAC weights file (%s): %s — using heuristic",
                path,
                exc,
            )
        except Exception as exc:
            log.error("Failed to load SAC weights: %s — using heuristic", exc)

        self._trained = False
        return False

    def reload_weights(self) -> bool:
        """Hot-reload weights after the offline trainer completes."""
        return self._load_weights()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_actor: SACActorNumpy | None = None


def get_actor() -> SACActorNumpy:
    global _actor
    if _actor is None:
        _actor = SACActorNumpy()
    return _actor


def compute_position_fraction(state: np.ndarray) -> float:
    """
    Public entry point — compute position-size fraction via the SAC actor.
    Called from brain.py (via run_in_executor in the async bot).

    Returns a float in (-1, 1):
        > 0  → long bias
        < 0  → short bias
        = 0  → flat
    """
    return get_actor().forward(state)


def reload_sac_weights() -> None:
    """Called after the offline trainer uploads new weights to the Pi."""
    actor = get_actor()
    actor.reload_weights()
    log.info("SAC V2 actor weights hot-reloaded")
