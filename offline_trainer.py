"""
offline_trainer.py
==================

Institutional-grade offline trainer for the SAC component of the live crypto
trading bot.

This module trains a Soft Actor-Critic (SAC) agent end-to-end with a
risk-adjusted, multi-component reward function and exports the actor weights
in a NumPy ``.npz`` format consumed by ``brain.py`` / ``rl_agent.py`` for
microsecond-level inference on the live Raspberry Pi 5 trading host.

Highlights
----------
*   STATE_DIM = 13 throughout (matches ``features.py`` and the live
    ``brain.compute_sac_state`` contract).
*   Composite reward built from seven institutionally-meaningful components:
    Differential Sharpe Ratio (Moody & Saffell, 1998), Sortino downside-
    deviation penalty, non-linear Maximum Drawdown penalty, retroactive
    "bad entry" penalty, time-decay holding cost, anti-paralysis idle
    penalty, and a realistic Binance Futures transaction cost model
    including ATR-scaled slippage.
*   Twin-Q critic SAC with automatic temperature tuning and Polyak target
    averaging.
*   Prioritized Experience Replay with proportional priorities and
    importance-sampling correction (annealed beta).
*   Custom Gym-compatible ``CryptoTradingEnv`` with HMM-style regime
    detection (TRENDING / RANGING / HIGH_VOL) and a curriculum that ramps
    episode length and reward complexity over the first 500 episodes.
*   Train/validation split (last 20% held out), with periodic out-of-sample
    validation episodes and an over-fitting alert when validation Sharpe
    drifts more than 0.5 below training Sharpe.
*   Auto-detects CUDA / CPU.  On CPU (e.g. the Pi) the trainer collapses
    its hidden widths and batch size to fit the host memory envelope.
*   SIGINT handler always flushes a checkpoint before exit.

Run standalone::

    python offline_trainer.py                       # full training run
    python offline_trainer.py --episodes 500        # short run
    python offline_trainer.py --resume              # continue from sac_full.pt
    python offline_trainer.py --validate-only       # eval saved actor only
    python offline_trainer.py --device cpu          # force CPU
    python offline_trainer.py --data my_data.csv    # custom data file

After training the file ``sac_actor.npz`` is dropped next to this script and
is immediately loadable by the live bot's ``rl_agent.SACActorNumpy``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for offline_trainer.py — install it with "
        "`pip install torch` on a laptop / dev box (the live Pi does not "
        "need PyTorch for inference)."
    ) from exc


# =====================================================================================
# LOGGING
# =====================================================================================

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("offline_trainer")


# =====================================================================================
# DIMENSION CONSTANTS
# =====================================================================================

# These two constants are the hard contract with brain.py / features.py / rl_agent.py.
# Do **not** change them here unless the live bot's state vector definition has been
# updated in lock-step.
STATE_DIM: int = 13
ACTION_DIM: int = 1


# =====================================================================================
# HYPERPARAMETER CONFIG BLOCKS
# =====================================================================================

SAC_CONFIG: dict[str, Any] = {
    # ── Network shape ────────────────────────────────────────────────────────────────
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "hidden_dims_gpu": [256, 256, 128],   # full-fat actor / critic on CUDA
    "hidden_dims_cpu": [128, 128, 64],    # collapsed widths to fit Pi memory budget

    # ── Optimization ─────────────────────────────────────────────────────────────────
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "grad_clip_norm": 1.0,
    "warmup_episodes": 100,
    "warmup_lr_scale": 0.1,

    # ── Replay buffer ────────────────────────────────────────────────────────────────
    "buffer_size": 500_000,
    "batch_size_gpu": 512,
    "batch_size_cpu": 128,
    "update_every": 4,
    "min_buffer_to_train": 2_000,

    # ── PER ──────────────────────────────────────────────────────────────────────────
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "per_eps": 1e-6,
    "per_priority_init": 1.0,

    # ── Stochastic actor / temperature ───────────────────────────────────────────────
    "log_std_min": -20.0,
    "log_std_max": 2.0,
    "init_alpha": 0.2,
    "target_entropy": -float(ACTION_DIM),
}


REWARD_CONFIG: dict[str, Any] = {
    # ── Top-level composite weights ──────────────────────────────────────────────────
    # Increase to make the agent more aggressive about chasing risk-adjusted equity
    # growth. Decreasing softens its bias toward online Sharpe at the expense of
    # absolute PnL.
    "w_sharpe": 1.0,

    # Increase to amplify the asymmetric downside-deviation penalty. A larger value
    # makes the agent very averse to clusters of negative steps without penalising
    # large winning streaks.
    "w_sortino": 0.5,

    # Increase to enforce stricter capital preservation. At very high values the
    # agent will sit in cash to avoid even moderate drawdowns; at low values the
    # agent tolerates deep equity excursions in pursuit of edge.
    "w_drawdown": 5.0,

    # Increase to make the agent more sensitive to mistimed entries. Larger values
    # train the agent to let setups mature instead of acting on noise.
    "w_bad_entry": 1.0,

    # Increase to discourage long holds. Higher values push the agent toward
    # faster trade resolution and away from "hold and hope" behaviour.
    "w_hold": 0.2,

    # Increase to discourage chronic abstention. Counteracts the trivial
    # "always-flat" minimum of risk-averse policies.
    "w_idle": 0.05,

    # ── Transaction cost model ───────────────────────────────────────────────────────
    # 0.06% per side, matching Binance Futures taker fees. Applied on every open
    # and close in price-fraction terms.
    "taker_fee": 0.0006,
    # Slippage in price-units = slippage_atr_mult * ATR * |size_fraction|.
    "slippage_atr_mult": 0.5,

    # ── Differential Sharpe (Moody & Saffell, 1998) ──────────────────────────────────
    # eta is the EWMA adaptation rate for the running first/second moments.
    "dsr_eta": 0.01,
    # Minimum allowable variance estimate; clamps to avoid division blow-ups
    # when only a handful of returns have been observed.
    "dsr_min_var": 1e-8,

    # ── Sortino penalty ──────────────────────────────────────────────────────────────
    "sortino_window": 200,
    "lambda_sortino": 0.5,

    # ── Drawdown penalty ─────────────────────────────────────────────────────────────
    "dd_threshold": 0.03,
    "lambda_dd": 1.0,

    # ── Bad-entry detection ──────────────────────────────────────────────────────────
    "bad_entry_atr_mult": 0.5,
    "bad_entry_lookback": 10,
    "lambda_bad_entry": 1.0,

    # ── Holding cost ─────────────────────────────────────────────────────────────────
    "max_hold_steps": 240,
    "lambda_hold": 1.0,

    # ── Idle / paralysis penalty ─────────────────────────────────────────────────────
    "idle_threshold": 0.05,
    "idle_k_steps": 20,
    "lambda_idle": 1.0,

    # ── Regime-specific lambda multipliers ───────────────────────────────────────────
    # Applied in addition to the top-level w_* weights when the env classifies the
    # current step into the corresponding regime. Mostly used to step up drawdown
    # penalty during HIGH_VOL bursts.
    "regime_high_vol_dd_mult": 1.5,
    "regime_high_vol_sharpe_mult": 0.7,
    "regime_trending_sharpe_mult": 1.2,
    "regime_ranging_hold_mult": 1.3,
}


TRAIN_CONFIG: dict[str, Any] = {
    "total_episodes": 2000,
    "episode_len_full": 2000,
    "episode_len_warmup": 500,
    "val_split": 0.20,
    "checkpoint_every": 100,
    "validate_every": 200,

    "log_path": "training_log.csv",
    "data_path": "training_data.csv",
    "actor_export_path": "sac_actor.npz",
    "checkpoint_path": "sac_full.pt",

    "curriculum_warmup_episodes": 200,
    "curriculum_ramp_end_episode": 500,

    "device_override": None,                 # {"cpu", "cuda", None=auto}
    "random_seed": 42,
    "validation_sharpe_alert_delta": 0.5,

    # Synthetic-data fallback parameters when no CSV is supplied.
    "synthetic_steps": 25_000,
    "synthetic_seed": 1337,
}


DATA_CONFIG: dict[str, Any] = {
    "csv_path": "training_data.csv",
    "feature_columns": [
        "price_change",
        "momentum",
        "atr_norm_vol",
        "volume_delta",
        "spread",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_position",
        "position_size",
        "position_pnl",
        "position_age",
    ],
    "price_col": "close",
    "atr_col": "atr",
    "timestamp_col": "timestamp",
}

assert len(DATA_CONFIG["feature_columns"]) == STATE_DIM, (
    "DATA_CONFIG feature column count must equal STATE_DIM"
)


# =====================================================================================
# UTILITY FUNCTIONS
# =====================================================================================

def set_global_seeds(seed: int) -> None:
    """
    Set every relevant RNG seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value applied to ``random``, ``numpy`` and ``torch`` (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(override: str | None) -> torch.device:
    """
    Decide between CUDA and CPU and emit a startup banner.

    Parameters
    ----------
    override : str or None
        "cpu" or "cuda" to force a device, or ``None`` to auto-detect.

    Returns
    -------
    torch.device
        The resolved device.  A warning is logged when CUDA is requested but
        unavailable.
    """
    if override is not None:
        choice = override.lower().strip()
        if choice not in {"cpu", "cuda"}:
            raise ValueError(f"--device must be cpu or cuda (got {override!r})")
        if choice == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU")
            return torch.device("cpu")
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning(
        "No CUDA device detected — running on CPU. Hidden widths and batch "
        "size will be reduced automatically to stay within Pi memory limits."
    )
    return torch.device("cpu")


def device_dependent_config(device: torch.device) -> tuple[list[int], int]:
    """
    Pick (hidden_dims, batch_size) based on the resolved device.

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    tuple[list[int], int]
        (hidden_dims, batch_size) tuned for the device.
    """
    if device.type == "cuda":
        return list(SAC_CONFIG["hidden_dims_gpu"]), int(SAC_CONFIG["batch_size_gpu"])
    return list(SAC_CONFIG["hidden_dims_cpu"]), int(SAC_CONFIG["batch_size_cpu"])


def log_memory_budget(device: torch.device) -> None:
    """Emit a startup banner describing the available memory budget."""
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info("Device: CUDA (%s) — %.2f GiB total VRAM", name, total_gb)
    else:
        try:
            import psutil  # type: ignore
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            logger.info("Device: CPU — %.2f GiB system RAM detected", total_gb)
        except ImportError:
            logger.info("Device: CPU (psutil not installed — memory budget unknown)")


# =====================================================================================
# DATA LOADING & SYNTHETIC FALLBACK
# =====================================================================================

def _generate_synthetic_dataset(n_steps: int, seed: int) -> pd.DataFrame:
    """
    Produce a synthetic OHLCV + feature dataset for smoke-testing.

    The generator builds a geometric-Brownian-motion price series with regime
    switches and computes the standard 13-feature vector on top.  This makes
    the trainer runnable with zero external data so a fresh checkout can
    smoke-test the full pipeline before the live CSV is plugged in.

    Parameters
    ----------
    n_steps : int
        Number of bars to generate.
    seed : int
        RNG seed.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the columns referenced by ``DATA_CONFIG``.
    """
    rng = np.random.default_rng(seed)

    base_drift = 0.00002
    base_vol = 0.0009
    regime_len = 800

    # Price walk with regime-switching drift / vol -----------------------------------
    drifts = np.zeros(n_steps)
    vols = np.zeros(n_steps)
    for start in range(0, n_steps, regime_len):
        end = min(start + regime_len, n_steps)
        d = base_drift * rng.choice([-1.5, -0.5, 0.5, 1.5])
        v = base_vol * rng.uniform(0.6, 2.5)
        drifts[start:end] = d
        vols[start:end] = v

    log_returns = drifts + vols * rng.standard_normal(n_steps)
    close = 30_000.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(rng.standard_normal(n_steps)) * vols * 0.5)
    low = close * (1.0 - np.abs(rng.standard_normal(n_steps)) * vols * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.lognormal(mean=8.0, sigma=0.6, size=n_steps)

    series = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)

    # ATR (Wilder, 14) ---------------------------------------------------------------
    pc = series.shift(1)
    tr = pd.concat(
        [high_s - low_s, (high_s - pc).abs(), (low_s - pc).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().bfill()

    # 13 features --------------------------------------------------------------------
    price_change = series.pct_change().fillna(0.0).values
    momentum = (series - series.shift(20)).fillna(0.0).values / np.maximum(close, 1e-9)
    atr_norm_vol = (atr / np.maximum(close, 1e-9)).values
    volume_delta = pd.Series(volume).pct_change().fillna(0.0).values
    spread = (high - low) / np.maximum(close, 1e-9)

    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))).fillna(50.0).values

    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).values
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = macd - macd_signal

    bb_mid = series.rolling(20).mean()
    bb_std = series.rolling(20).std()
    upper = bb_mid + 2 * bb_std
    lower = bb_mid - 2 * bb_std
    bb_position = ((series - lower) / (upper - lower + 1e-9)).fillna(0.5).values

    # Position metadata is set live by the env at step time; the CSV columns just
    # need to exist with placeholder zeros.
    n = n_steps
    position_size = np.zeros(n)
    position_pnl = np.zeros(n)
    position_age = np.zeros(n)

    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr.values,
            "price_change": price_change,
            "momentum": momentum,
            "atr_norm_vol": atr_norm_vol,
            "volume_delta": volume_delta,
            "spread": spread,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_position": bb_position,
            "position_size": position_size,
            "position_pnl": position_pnl,
            "position_age": position_age,
        }
    )
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the training CSV or fall back to a synthetic dataset.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV.  If the file does not exist a deterministic synthetic
        dataset is generated so the pipeline still runs end-to-end.

    Returns
    -------
    pandas.DataFrame
        DataFrame ready for ``CryptoTradingEnv``.
    """
    path = Path(csv_path)
    if path.exists():
        logger.info("Loading dataset from %s", path.resolve())
        df = pd.read_csv(path)
        required = (
            list(DATA_CONFIG["feature_columns"])
            + [DATA_CONFIG["price_col"], DATA_CONFIG["atr_col"]]
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV {path} is missing required columns: {missing}")
        return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    logger.warning(
        "Dataset CSV %s not found — generating synthetic %d-step dataset for "
        "smoke testing. Provide --data <path> for a real run.",
        path,
        TRAIN_CONFIG["synthetic_steps"],
    )
    return _generate_synthetic_dataset(
        n_steps=int(TRAIN_CONFIG["synthetic_steps"]),
        seed=int(TRAIN_CONFIG["synthetic_seed"]),
    )


# =====================================================================================
# RUNNING STANDARD SCALER (NO LOOK-AHEAD)
# =====================================================================================

class RunningStandardScaler:
    """
    Online StandardScaler fitted only on a designated training slice.

    Notes
    -----
    The scaler is fit once on ``[0, train_end)`` and frozen.  Validation steps
    are transformed but never used to update statistics — preventing forward
    data leakage into the agent's normalization.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fitted_: bool = False

    def fit(self, x: np.ndarray) -> "RunningStandardScaler":
        """
        Compute and freeze the mean/std on the supplied feature matrix.

        Parameters
        ----------
        x : numpy.ndarray
            (n_samples, n_features) matrix.

        Returns
        -------
        RunningStandardScaler
            ``self`` for chaining.
        """
        if x.ndim != 2:
            raise ValueError("Scaler expects a 2-D feature matrix")
        self.mean_ = x.mean(axis=0).astype(np.float32)
        self.std_ = x.std(axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        self.fitted_ = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the frozen mean/std to ``x`` and return ``float32``."""
        if not self.fitted_:
            raise RuntimeError("Scaler must be fit before transform")
        out = (x - self.mean_) / self.std_
        return out.astype(np.float32)


# =====================================================================================
# REGIME DETECTOR
# =====================================================================================

class RegimeDetector:
    """
    Lightweight HMM-style regime classifier.

    Each step is labeled one of {``TRENDING``, ``RANGING``, ``HIGH_VOL``}
    based on a rolling ATR percentile and recent price momentum.
    """

    REGIME_TRENDING = "TRENDING"
    REGIME_RANGING = "RANGING"
    REGIME_HIGH_VOL = "HIGH_VOL"

    def __init__(
        self,
        atr_percentile_window: int = 500,
        momentum_window: int = 30,
        high_vol_pct: float = 0.85,
        trending_momentum: float = 0.005,
    ) -> None:
        """
        Parameters
        ----------
        atr_percentile_window : int
            Rolling window for ATR percentile.
        momentum_window : int
            Window for the price momentum diagnostic.
        high_vol_pct : float
            ATR percentile above which we declare HIGH_VOL.
        trending_momentum : float
            Absolute log-return-per-window threshold for TRENDING.
        """
        self.atr_window = atr_percentile_window
        self.mom_window = momentum_window
        self.high_vol_pct = high_vol_pct
        self.trending_momentum = trending_momentum

    def classify(
        self,
        atr_series: np.ndarray,
        close_series: np.ndarray,
        idx: int,
    ) -> str:
        """
        Return the regime label for index ``idx``.

        Parameters
        ----------
        atr_series : numpy.ndarray
            Full ATR vector.
        close_series : numpy.ndarray
            Full close-price vector.
        idx : int
            Index inside ``atr_series`` / ``close_series`` to classify.

        Returns
        -------
        str
            One of ``"TRENDING"``, ``"RANGING"``, ``"HIGH_VOL"``.
        """
        lo = max(0, idx - self.atr_window)
        atr_window = atr_series[lo: idx + 1]
        if atr_window.size < 5:
            return self.REGIME_RANGING
        cur_atr = atr_series[idx]
        # Rank-based percentile (robust to outliers).
        pct = float(np.mean(atr_window <= cur_atr))
        if pct >= self.high_vol_pct:
            return self.REGIME_HIGH_VOL
        mom_lo = max(0, idx - self.mom_window)
        denom = max(close_series[mom_lo], 1e-9)
        mom = math.log(max(close_series[idx], 1e-9) / denom)
        if abs(mom) >= self.trending_momentum:
            return self.REGIME_TRENDING
        return self.REGIME_RANGING


# =====================================================================================
# REWARD CALCULATOR — INSTITUTIONAL COMPOSITE
# =====================================================================================

@dataclass
class TradeRecord:
    """In-flight book-keeping for the most recent open position."""

    entry_step: int
    entry_price: float
    size_fraction: float
    entry_atr: float
    pnl_path: list[float] = field(default_factory=list)


class RewardCalculator:
    """
    Computes the seven-component composite step reward.

    Components
    ----------
    1.  Differential Sharpe Ratio (Moody & Saffell, 1998) — dense, online,
        risk-adjusted feedback.
    2.  Sortino downside-deviation penalty over a rolling window.
    3.  Continuous, non-linear maximum drawdown penalty (squared above
        threshold).
    4.  Retroactive bad-entry penalty triggered when fresh longs immediately
        give back >0.5 * ATR.
    5.  Time-decay holding cost growing as ``(steps_in_trade / max_hold)^2``.
    6.  Idle penalty when the agent abstains (``|action| < idle_threshold``)
        for K consecutive steps.
    7.  Realistic Binance-Futures transaction costs: taker fee + ATR-scaled
        slippage on every open and close.

    The composite reward at step ``t`` is::

        R_t = w1 * dSR_t
            - w2 * sortino_penalty
            - w3 * dd_penalty
            - w4 * bad_entry_penalty
            - w5 * hold_penalty
            - w6 * idle_penalty
            - transaction_cost

    A regime-aware multiplier may further re-scale the drawdown / Sharpe /
    hold contributions based on the current regime label.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        config : dict
            ``REWARD_CONFIG``-style dict (already merged with any overrides).
        """
        self.cfg = config

        # DSR running estimates ------------------------------------------------------
        self.dsr_A: float = 0.0
        self.dsr_B: float = 0.0
        self.dsr_initialized: bool = False

        # Sortino rolling window -----------------------------------------------------
        self.return_window: deque[float] = deque(maxlen=int(config["sortino_window"]))

        # Equity peak for drawdown ---------------------------------------------------
        self.peak_equity: float = 1.0
        self.equity: float = 1.0

        # Idle counter ---------------------------------------------------------------
        self.idle_steps: int = 0

        # Bad-entry book-keeping -----------------------------------------------------
        self.active_trade: TradeRecord | None = None
        self._pending_bad_entry_penalty: float = 0.0

        # Curriculum scaling (set by env at episode start) ---------------------------
        self.curriculum_scale: float = 1.0

        # Episode-level diagnostics --------------------------------------------------
        self.episode_returns: list[float] = []
        self.regime_counts: dict[str, int] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────────────
    def reset(self, curriculum_scale: float = 1.0) -> None:
        """
        Reset all running state at the start of an episode.

        Parameters
        ----------
        curriculum_scale : float
            Value in ``[0, 1]`` controlling how many of the auxiliary penalty
            components are active.  ``0`` enables only DSR + tx cost (warmup);
            ``1`` enables every component (full reward).
        """
        self.dsr_A = 0.0
        self.dsr_B = 0.0
        self.dsr_initialized = False
        self.return_window.clear()
        self.peak_equity = 1.0
        self.equity = 1.0
        self.idle_steps = 0
        self.active_trade = None
        self._pending_bad_entry_penalty = 0.0
        self.curriculum_scale = float(max(0.0, min(1.0, curriculum_scale)))
        self.episode_returns = []
        self.regime_counts = {}

    # ── Trade plumbing ───────────────────────────────────────────────────────────────
    def open_trade(
        self,
        step: int,
        price: float,
        size_fraction: float,
        atr: float,
    ) -> float:
        """
        Record a fresh trade and return the realized transaction cost.

        Returns
        -------
        float
            The transaction-cost magnitude (always >= 0).  Already in equity
            (return) units — i.e. fractional cost on a unit notional.
        """
        self.active_trade = TradeRecord(
            entry_step=step,
            entry_price=float(price),
            size_fraction=float(size_fraction),
            entry_atr=float(atr),
            pnl_path=[],
        )
        return self._tx_cost(size_fraction=abs(size_fraction), atr=atr, price=price)

    def close_trade(self, price: float, atr: float) -> float:
        """
        Close the active trade and return the realized transaction cost.
        """
        if self.active_trade is None:
            return 0.0
        size = abs(self.active_trade.size_fraction)
        self.active_trade = None
        return self._tx_cost(size_fraction=size, atr=atr, price=price)

    def _tx_cost(self, size_fraction: float, atr: float, price: float) -> float:
        """Combined fee + slippage for a single open or close."""
        fee = float(self.cfg["taker_fee"]) * size_fraction
        slip_price = float(self.cfg["slippage_atr_mult"]) * float(atr) * size_fraction
        slip_frac = slip_price / max(price, 1e-9)
        return float(fee + slip_frac)

    # ── Step return / equity update ──────────────────────────────────────────────────
    def update_step_return(self, step_return: float) -> None:
        """
        Push a fresh step return into all running statistics.

        Parameters
        ----------
        step_return : float
            Net return for this step (already after transaction costs).  Used
            to update DSR, Sortino, equity peak, drawdown.
        """
        sr = float(step_return)
        self.episode_returns.append(sr)
        self.return_window.append(sr)
        self.equity *= (1.0 + sr)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Roll the bad-entry pnl path forward.
        if self.active_trade is not None:
            self.active_trade.pnl_path.append(sr)

    # ── Component computations ───────────────────────────────────────────────────────
    def differential_sharpe(self, step_return: float) -> float:
        """
        Compute the Moody-Saffell differential Sharpe contribution.

        Notes
        -----
        Uses EWMA estimates of mean (``A``) and mean-square (``B``) return
        with adaptation rate ``eta``.  The contribution at step ``t`` is::

            delta_A   = R_t - A_{t-1}
            delta_B   = R_t**2 - B_{t-1}
            denom     = (B_{t-1} - A_{t-1}**2) ** 1.5
            dSR_t     = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / denom

        After the contribution is computed the EWMA estimates are updated.
        """
        eta = float(self.cfg["dsr_eta"])
        min_var = float(self.cfg["dsr_min_var"])
        r = float(step_return)

        if not self.dsr_initialized:
            # First step: bootstrap A / B and emit zero contribution.
            self.dsr_A = r
            self.dsr_B = r * r
            self.dsr_initialized = True
            return 0.0

        delta_A = r - self.dsr_A
        delta_B = r * r - self.dsr_B
        var = self.dsr_B - self.dsr_A * self.dsr_A
        var = max(var, min_var)
        denom = var ** 1.5
        dsr = (self.dsr_B * delta_A - 0.5 * self.dsr_A * delta_B) / max(denom, 1e-12)

        # Online update of A, B for the next step.
        self.dsr_A = self.dsr_A + eta * delta_A
        self.dsr_B = self.dsr_B + eta * delta_B

        # Clamp to a sane range — DSR can spike on the first ~10 steps.
        return float(max(-10.0, min(10.0, dsr)))

    def sortino_penalty(self) -> float:
        """
        Downside-deviation penalty (returns >= 0).

        Returns
        -------
        float
            ``lambda_sortino * sqrt(mean(min(r, 0)^2))`` over the rolling
            window.  Always non-negative.
        """
        if len(self.return_window) < 5:
            return 0.0
        arr = np.fromiter(self.return_window, dtype=np.float64)
        downside = np.minimum(arr, 0.0)
        dd = float(np.sqrt(np.mean(downside * downside)))
        return float(self.cfg["lambda_sortino"]) * dd

    def drawdown_penalty(self, regime: str) -> float:
        """
        Continuous non-linear drawdown penalty.

        Parameters
        ----------
        regime : str
            Current regime label.  HIGH_VOL boosts the penalty by
            ``regime_high_vol_dd_mult``.
        """
        peak = max(self.peak_equity, 1e-9)
        dd = (peak - self.equity) / peak
        threshold = float(self.cfg["dd_threshold"])
        excess = max(0.0, dd - threshold)
        base = float(self.cfg["lambda_dd"]) * (excess ** 2)
        if regime == RegimeDetector.REGIME_HIGH_VOL:
            base *= float(self.cfg["regime_high_vol_dd_mult"])
        return float(base)

    def hold_penalty(self, current_step: int, regime: str) -> float:
        """
        Quadratic time-decay holding cost on the active trade.
        """
        if self.active_trade is None:
            return 0.0
        steps_in_trade = max(0, current_step - self.active_trade.entry_step)
        ratio = steps_in_trade / float(max(1, self.cfg["max_hold_steps"]))
        base = float(self.cfg["lambda_hold"]) * (ratio ** 2)
        if regime == RegimeDetector.REGIME_RANGING:
            base *= float(self.cfg["regime_ranging_hold_mult"])
        return float(base)

    def idle_penalty(self, action: float) -> float:
        """
        Anti-paralysis penalty when |action| stays below ``idle_threshold``
        for K consecutive steps.
        """
        if abs(action) < float(self.cfg["idle_threshold"]):
            self.idle_steps += 1
        else:
            self.idle_steps = 0
        if self.idle_steps <= int(self.cfg["idle_k_steps"]):
            return 0.0
        excess = self.idle_steps - int(self.cfg["idle_k_steps"])
        return float(self.cfg["lambda_idle"]) * math.log1p(excess)

    def bad_entry_penalty(self, current_step: int) -> float:
        """
        Retroactive penalty applied within the lookback window when a
        freshly-opened trade gives back > ``bad_entry_atr_mult`` * ATR.
        """
        if self.active_trade is None:
            return 0.0
        steps_in_trade = current_step - self.active_trade.entry_step
        if steps_in_trade <= 0 or steps_in_trade > int(self.cfg["bad_entry_lookback"]):
            return 0.0
        # Cumulative log-return path equivalent (sum of step returns).
        unrealized_frac = float(np.sum(self.active_trade.pnl_path))
        unrealized_price_move = unrealized_frac * self.active_trade.entry_price
        # If position is long (size > 0), bad entry = strongly negative move.
        signed = unrealized_price_move * math.copysign(1.0, self.active_trade.size_fraction)
        threshold = float(self.cfg["bad_entry_atr_mult"]) * self.active_trade.entry_atr
        if signed >= -threshold:
            return 0.0
        # The deeper the giveback, the larger the penalty (clipped).
        magnitude = min((abs(signed) - threshold) / max(threshold, 1e-9), 5.0)
        return float(self.cfg["lambda_bad_entry"]) * magnitude

    # ── Composite ───────────────────────────────────────────────────────────────────
    def compose(
        self,
        step: int,
        action: float,
        step_return: float,
        regime: str,
        transaction_cost: float,
    ) -> tuple[float, dict[str, float]]:
        """
        Combine all seven components into the scalar step reward.

        Parameters
        ----------
        step : int
            Current step within the episode.
        action : float
            Agent's continuous action (sac_fraction in ``[-1, 1]``).
        step_return : float
            Net return for this step (already includes transaction costs).
        regime : str
            Regime label from ``RegimeDetector``.
        transaction_cost : float
            Realized open/close fee+slippage cost for this step (>= 0).

        Returns
        -------
        tuple[float, dict[str, float]]
            (composite_reward, breakdown_dict).
        """
        cs = self.curriculum_scale
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1

        # 1. DSR
        dsr = self.differential_sharpe(step_return)

        # 2-6. Penalties
        sortino_p = self.sortino_penalty() * cs
        dd_p = self.drawdown_penalty(regime) * cs
        bad_p = self.bad_entry_penalty(step) * cs
        hold_p = self.hold_penalty(step, regime) * cs
        idle_p = self.idle_penalty(action) * cs

        # Regime modulation of Sharpe weight.
        w_sharpe = float(self.cfg["w_sharpe"] if "w_sharpe" in self.cfg else REWARD_CONFIG["w_sharpe"])
        if regime == RegimeDetector.REGIME_HIGH_VOL:
            w_sharpe *= float(self.cfg["regime_high_vol_sharpe_mult"])
        elif regime == RegimeDetector.REGIME_TRENDING:
            w_sharpe *= float(self.cfg["regime_trending_sharpe_mult"])

        composite = (
            w_sharpe * dsr
            - float(self.cfg["w_sortino"]) * sortino_p
            - float(self.cfg["w_drawdown"]) * dd_p
            - float(self.cfg["w_bad_entry"]) * bad_p
            - float(self.cfg["w_hold"]) * hold_p
            - float(self.cfg["w_idle"]) * idle_p
            - float(transaction_cost)
        )
        # Numerical guardrail: gradients explode if the reward goes to ±inf.
        composite = float(max(-50.0, min(50.0, composite)))

        breakdown = {
            "dsr": float(dsr),
            "sortino": float(sortino_p),
            "drawdown": float(dd_p),
            "bad_entry": float(bad_p),
            "hold": float(hold_p),
            "idle": float(idle_p),
            "tx_cost": float(transaction_cost),
            "composite": composite,
        }
        return composite, breakdown


# =====================================================================================
# CRYPTO TRADING ENVIRONMENT
# =====================================================================================

class CryptoTradingEnv:
    """
    Gym-compatible environment for SAC training on the live bot's 13-feature state.

    Notes
    -----
    *   Long / flat only (matches ``bot.py`` live behavior).
    *   ``sac_fraction > 0.1`` opens (or holds) a long; ``<= 0.1`` exits / stays flat.
    *   Episode = randomly sampled contiguous window of (curriculum-scaled) length.
    *   Live regime detection on each step.
    *   Curriculum simplifies reward and shortens window before episode 200.
    """

    OBSERVATION_DIM: int = STATE_DIM
    ACTION_LOW: float = -1.0
    ACTION_HIGH: float = 1.0

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: RunningStandardScaler,
        reward_calc: RewardCalculator,
        regime_detector: RegimeDetector,
        train_end_idx: int,
        rng: np.random.Generator,
        config: dict[str, Any],
    ) -> None:
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Full dataset (including val tail).
        scaler : RunningStandardScaler
            Pre-fit scaler (training slice only).
        reward_calc : RewardCalculator
            Composite reward computer.
        regime_detector : RegimeDetector
            Live regime classifier.
        train_end_idx : int
            Index where the training slice ends (validation begins).
        rng : numpy.random.Generator
            Episode RNG.
        config : dict
            Training config (``TRAIN_CONFIG``-style).
        """
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.reward_calc = reward_calc
        self.regime_detector = regime_detector
        self.train_end_idx = train_end_idx
        self.rng = rng
        self.cfg = config

        feats = list(DATA_CONFIG["feature_columns"])
        self.feature_matrix = self.df[feats].values.astype(np.float32)
        self.scaled_features = self.scaler.transform(self.feature_matrix)
        self.close = self.df[DATA_CONFIG["price_col"]].values.astype(np.float64)
        self.atr = self.df[DATA_CONFIG["atr_col"]].values.astype(np.float64)

        # Episode state -----------------------------------------------------------
        self.episode_idx: int = 0
        self.start_idx: int = 0
        self.end_idx: int = 0
        self.current_idx: int = 0
        self.episode_len: int = int(self.cfg["episode_len_full"])
        self.position_open: bool = False
        self.position_size: float = 0.0
        self.position_entry_price: float = 0.0
        self.position_age: int = 0

        # Diagnostics -------------------------------------------------------------
        self.trade_returns: list[float] = []
        self.trade_durations: list[int] = []
        self.trades_executed: int = 0
        self.wins: int = 0
        self.last_trade_entry_step: int | None = None

    # ── Curriculum helpers ───────────────────────────────────────────────────────────
    def _curriculum_episode_len(self) -> int:
        """Episode length grows linearly from warmup → full over the ramp."""
        warmup = int(self.cfg["curriculum_warmup_episodes"])
        ramp_end = int(self.cfg["curriculum_ramp_end_episode"])
        full = int(self.cfg["episode_len_full"])
        short = int(self.cfg["episode_len_warmup"])
        if self.episode_idx < warmup:
            return short
        if self.episode_idx >= ramp_end:
            return full
        prog = (self.episode_idx - warmup) / max(1, ramp_end - warmup)
        return int(short + prog * (full - short))

    def _curriculum_scale(self) -> float:
        """Auxiliary-component activation grows linearly from 0 → 1."""
        warmup = int(self.cfg["curriculum_warmup_episodes"])
        ramp_end = int(self.cfg["curriculum_ramp_end_episode"])
        if self.episode_idx < warmup:
            return 0.0
        if self.episode_idx >= ramp_end:
            return 1.0
        return (self.episode_idx - warmup) / max(1, ramp_end - warmup)

    # ── Reset / step ─────────────────────────────────────────────────────────────────
    def reset(
        self,
        episode_idx: int,
        validation: bool = False,
    ) -> np.ndarray:
        """
        Sample a new contiguous window and return the initial state.

        Parameters
        ----------
        episode_idx : int
            Global episode counter (used by curriculum).
        validation : bool
            If True the start is sampled from the held-out tail and the
            curriculum is bypassed.

        Returns
        -------
        numpy.ndarray
            Initial scaled state of length 13.
        """
        self.episode_idx = int(episode_idx)
        if validation:
            self.episode_len = int(self.cfg["episode_len_full"])
            curric = 1.0
            lo = self.train_end_idx
            hi = max(lo + 1, len(self.df) - self.episode_len - 1)
            if hi <= lo + 1:
                self.start_idx = lo
            else:
                self.start_idx = int(self.rng.integers(lo, hi))
        else:
            self.episode_len = self._curriculum_episode_len()
            curric = self._curriculum_scale()
            lo = 100  # leave headroom for indicator warmup
            hi = max(lo + 1, self.train_end_idx - self.episode_len - 1)
            if hi <= lo + 1:
                self.start_idx = lo
            else:
                self.start_idx = int(self.rng.integers(lo, hi))

        self.end_idx = min(self.start_idx + self.episode_len, len(self.df) - 1)
        self.current_idx = self.start_idx

        # Reset book-keeping ---------------------------------------------------
        self.position_open = False
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_age = 0
        self.trade_returns.clear()
        self.trade_durations.clear()
        self.trades_executed = 0
        self.wins = 0
        self.last_trade_entry_step = None
        self.reward_calc.reset(curriculum_scale=curric)

        return self._build_state()

    def _build_state(self) -> np.ndarray:
        """Construct the 13-feature scaled state for the current index."""
        base = self.scaled_features[self.current_idx].copy()
        # Override the three "position metadata" fields live so the agent
        # observes its own position even when the CSV has them zeroed.
        base[-3] = np.float32(self.position_size)
        base[-2] = np.float32(self._unrealized_pnl_fraction())
        base[-1] = np.float32(self.position_age / 100.0)
        return base.astype(np.float32)

    def _unrealized_pnl_fraction(self) -> float:
        if not self.position_open or self.position_entry_price <= 0:
            return 0.0
        cur_price = float(self.close[self.current_idx])
        return (cur_price - self.position_entry_price) / self.position_entry_price

    def step(
        self,
        action: float,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Advance the env one bar.

        Parameters
        ----------
        action : float
            Continuous SAC action in ``[-1, 1]``.

        Returns
        -------
        next_state : numpy.ndarray
        reward : float
        done : bool
        info : dict
            Diagnostics: regime, position state, raw step return, reward
            breakdown, transaction cost.
        """
        action_clipped = float(np.clip(action, self.ACTION_LOW, self.ACTION_HIGH))
        prev_idx = self.current_idx
        next_idx = min(prev_idx + 1, self.end_idx)

        cur_price = float(self.close[prev_idx])
        next_price = float(self.close[next_idx])
        cur_atr = float(self.atr[prev_idx])
        regime = self.regime_detector.classify(self.atr, self.close, prev_idx)

        # Decide trade event ---------------------------------------------------
        tx_cost = 0.0
        opened_this_step = False
        closed_this_step = False
        size_used = self.position_size

        if not self.position_open and action_clipped > 0.1:
            self.position_open = True
            self.position_size = float(action_clipped)
            self.position_entry_price = cur_price
            self.position_age = 0
            self.last_trade_entry_step = prev_idx
            tx_cost += self.reward_calc.open_trade(
                step=prev_idx,
                price=cur_price,
                size_fraction=self.position_size,
                atr=cur_atr,
            )
            opened_this_step = True
            size_used = self.position_size
        elif self.position_open and action_clipped <= 0.1:
            log_ret = math.log(max(next_price, 1e-12) / max(cur_price, 1e-12))
            realized = self.position_size * (math.exp(log_ret) - 1.0)
            tx_cost += self.reward_calc.close_trade(price=cur_price, atr=cur_atr)
            self.trade_returns.append(realized - tx_cost)
            duration = (
                prev_idx - (self.last_trade_entry_step or prev_idx)
                if self.last_trade_entry_step is not None
                else 0
            )
            self.trade_durations.append(int(max(0, duration)))
            self.trades_executed += 1
            if realized - tx_cost > 0:
                self.wins += 1
            self.position_open = False
            size_used = self.position_size
            self.position_size = 0.0
            self.position_entry_price = 0.0
            self.position_age = 0
            self.last_trade_entry_step = None
            closed_this_step = True

        # Step return on the held position ------------------------------------
        if self.position_open:
            log_ret = math.log(max(next_price, 1e-12) / max(cur_price, 1e-12))
            price_step_return = self.position_size * (math.exp(log_ret) - 1.0)
            self.position_age += 1
        else:
            price_step_return = 0.0

        net_step_return = price_step_return - tx_cost
        self.reward_calc.update_step_return(net_step_return)

        reward, breakdown = self.reward_calc.compose(
            step=prev_idx,
            action=action_clipped,
            step_return=net_step_return,
            regime=regime,
            transaction_cost=tx_cost,
        )

        # Advance index --------------------------------------------------------
        self.current_idx = next_idx
        done = self.current_idx >= self.end_idx
        next_state = self._build_state()

        info: dict[str, Any] = {
            "regime": regime,
            "position_open": self.position_open,
            "position_size": self.position_size,
            "step_return": net_step_return,
            "tx_cost": tx_cost,
            "opened": opened_this_step,
            "closed": closed_this_step,
            "size_used": size_used,
            "reward_breakdown": breakdown,
        }
        return next_state, float(reward), bool(done), info

    # ── Episode-level diagnostics ────────────────────────────────────────────────────
    def episode_metrics(self) -> dict[str, float]:
        """Compute end-of-episode performance metrics."""
        rets = np.asarray(self.reward_calc.episode_returns, dtype=np.float64)
        if rets.size == 0:
            return {
                "net_pnl": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_dd": 0.0,
                "win_rate": 0.0,
                "avg_trade_duration": 0.0,
                "total_trades": 0.0,
            }
        equity = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-12)
        max_dd = float(np.max(dd))
        net_pnl = float(equity[-1] - 1.0)
        mu = float(np.mean(rets))
        sd = float(np.std(rets))
        ann = math.sqrt(525_600)  # 1-minute bars -> annualized
        sharpe = (mu / max(sd, 1e-12)) * ann
        downside = rets[rets < 0]
        ds = float(np.sqrt(np.mean(downside ** 2))) if downside.size else 0.0
        sortino = (mu / max(ds, 1e-12)) * ann if ds > 0 else 0.0
        calmar = net_pnl / max(max_dd, 1e-12) if max_dd > 0 else 0.0
        wr = self.wins / max(1, self.trades_executed)
        avg_dur = float(np.mean(self.trade_durations)) if self.trade_durations else 0.0
        return {
            "net_pnl": net_pnl,
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_dd": max_dd,
            "win_rate": float(wr),
            "avg_trade_duration": avg_dur,
            "total_trades": float(self.trades_executed),
        }


# =====================================================================================
# PRIORITIZED EXPERIENCE REPLAY (PROPORTIONAL)
# =====================================================================================

class PrioritizedReplayBuffer:
    """
    Proportional-priority replay buffer with importance-sampling correction.

    Stores up to ``capacity`` ``(state, action, reward, next_state, done)``
    transitions.  Sampling probability is proportional to
    ``priority ** alpha`` and IS weights ``(1 / (N * P)) ** beta`` are
    returned alongside the batch for unbiased gradient updates.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float,
        eps: float,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float64)

        self.ptr: int = 0
        self.size: int = 0
        self.max_priority: float = float(SAC_CONFIG["per_priority_init"])

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Insert a transition, assigning the current max priority."""
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i, 0] = float(reward)
        self.next_states[i] = next_state
        self.dones[i, 0] = float(bool(done))
        self.priorities[i] = self.max_priority
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        """
        Sample a batch with importance-sampling weights.

        Returns
        -------
        tuple
            (states, actions, rewards, next_states, dones, weights, indices).
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        prios = self.priorities[: self.size] ** self.alpha
        prob_sum = prios.sum()
        if prob_sum <= 0 or not np.isfinite(prob_sum):
            probs = np.full(self.size, 1.0 / self.size, dtype=np.float64)
        else:
            probs = prios / prob_sum
        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= max(weights.max(), 1e-12)
        weights = weights.astype(np.float32)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.reshape(-1, 1),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities given fresh TD errors."""
        new = (np.abs(td_errors) + self.eps).astype(np.float64).flatten()
        self.priorities[indices] = new
        if new.size:
            self.max_priority = max(self.max_priority, float(new.max()))


# =====================================================================================
# SAC NETWORKS
# =====================================================================================

def _mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a stack of ``Linear`` + activation layers terminating in ``out_dim``."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SACActor(nn.Module):
    """
    Stochastic squashed-Gaussian actor.

    Architecture
    ------------
    ``[state_dim] -> hidden_dims -> (mu_head, log_std_head) -> tanh squash``

    The forward pass emits the deterministic ``mu`` (used for inference and
    weight export), and ``sample()`` returns a reparameterised action plus
    its log-prob with the tanh-Jacobian correction.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Shared trunk -------------------------------------------------------
        trunk: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            trunk.append(nn.Linear(prev, h))
            trunk.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk)
        # Heads --------------------------------------------------------------
        self.mu_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, log_std)`` heads, with ``log_std`` clamped."""
        h = self.trunk(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterised sample.

        Returns
        -------
        action : torch.Tensor
            Tanh-squashed action in ``(-1, 1)``.
        log_prob : torch.Tensor
            Log-prob with tanh Jacobian correction.
        mu_squashed : torch.Tensor
            Deterministic ``tanh(mu)`` (used for inference / export).
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        # Jacobian correction: log(1 - tanh(z)^2) — Soft Actor Critic Eq. 26.
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class SACCritic(nn.Module):
    """Twin Q-networks ``Q1, Q2: S × A -> R`` with min-clipping at the call site."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, hidden_dims, 1)
        self.q2 = _mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# =====================================================================================
# SAC AGENT
# =====================================================================================

class SACAgent:
    """
    Soft Actor-Critic agent with twin critics, automatic temperature tuning,
    Polyak-averaged targets, and gradient clipping.

    Loss formulation matches Haarnoja et al. 2018 with the entropy-regularized
    Bellman backup::

        y = r + gamma * (1 - done) * (min_i Q_target_i(s', a') - alpha * log_pi(a'|s'))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        self.cfg = config
        self.device = device

        self.actor = SACActor(
            state_dim,
            action_dim,
            hidden_dims,
            float(config["log_std_min"]),
            float(config["log_std_max"]),
        ).to(device)
        self.critic = SACCritic(state_dim, action_dim, hidden_dims).to(device)
        self.target_critic = SACCritic(state_dim, action_dim, hidden_dims).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=float(config["lr_actor"]))
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=float(config["lr_critic"]))

        # Learnable log-alpha for automatic entropy tuning.
        init_alpha = float(config["init_alpha"])
        self.log_alpha = torch.tensor(
            math.log(max(init_alpha, 1e-6)), device=device, requires_grad=True
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=float(config["lr_alpha"]))
        self.target_entropy = float(config["target_entropy"])

        self.tau = float(config["tau"])
        self.gamma = float(config["gamma"])
        self.grad_clip = float(config["grad_clip_norm"])

        self.last_alpha: float = init_alpha
        self.last_actor_loss: float = 0.0
        self.last_critic_loss: float = 0.0

    # ── Action selection ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> float:
        """
        Sample an action for env interaction.

        Parameters
        ----------
        state : numpy.ndarray
            13-feature state vector.
        deterministic : bool
            If True, return ``tanh(mu)`` (used for validation / live export).
        """
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if deterministic:
            mu, _ = self.actor(s)
            action = torch.tanh(mu)
        else:
            action, _, _ = self.actor.sample(s)
        return float(action.cpu().numpy().flatten()[0])

    # ── Update step ──────────────────────────────────────────────────────────────────
    def update(
        self,
        buffer: PrioritizedReplayBuffer,
        batch_size: int,
        beta: float,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Run a single gradient step on actor + twin critics + alpha.

        Returns
        -------
        tuple
            (actor_loss, critic_loss, alpha, indices, td_errors)
        """
        states_np, acts_np, rews_np, nexts_np, dones_np, weights_np, idxs = (
            buffer.sample(batch_size=batch_size, beta=beta)
        )
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(acts_np).to(self.device)
        rewards = torch.from_numpy(rews_np).to(self.device)
        nexts = torch.from_numpy(nexts_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        weights = torch.from_numpy(weights_np).to(self.device)

        alpha = self.log_alpha.exp().detach()

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(nexts)
            tq1, tq2 = self.target_critic(nexts, next_actions)
            target_q = torch.min(tq1, tq2) - alpha * next_log_pi
            backup = rewards + self.gamma * (1.0 - dones) * target_q

        q1, q2 = self.critic(states, actions)
        td1 = q1 - backup
        td2 = q2 - backup
        critic_loss = (weights * (td1.pow(2) + td2.pow(2))).mean()

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        # ── Actor update ───────────────────────────────────────────────────
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = ((alpha * log_pi) - q_new).mean()
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        # ── Alpha update ──────────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Polyak target update ──────────────────────────────────────────
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        # Cache last metrics for logging.
        self.last_actor_loss = float(actor_loss.detach().item())
        self.last_critic_loss = float(critic_loss.detach().item())
        self.last_alpha = float(self.log_alpha.exp().detach().item())

        td_errors = (0.5 * (td1.detach().abs() + td2.detach().abs())).cpu().numpy().flatten()
        return self.last_actor_loss, self.last_critic_loss, self.last_alpha, idxs, td_errors

    # ── Learning-rate scheduling ────────────────────────────────────────────────────
    def set_lr_scale(self, scale: float) -> None:
        """Apply a global multiplicative scale to all three optimizers."""
        scale = float(max(1e-6, scale))
        for pg in self.actor_optim.param_groups:
            pg["lr"] = float(self.cfg["lr_actor"]) * scale
        for pg in self.critic_optim.param_groups:
            pg["lr"] = float(self.cfg["lr_critic"]) * scale
        for pg in self.alpha_optim.param_groups:
            pg["lr"] = float(self.cfg["lr_alpha"]) * scale

    # ── Checkpointing ────────────────────────────────────────────────────────────────
    def save_full(self, path: str | Path, episode: int) -> None:
        """Save a full PyTorch checkpoint suitable for ``--resume``."""
        torch.save(
            {
                "episode": int(episode),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optim": self.alpha_optim.state_dict(),
                "hidden_dims": list(self.actor.hidden_dims),
                "state_dim": int(self.actor.state_dim),
                "action_dim": int(self.actor.action_dim),
            },
            str(path),
        )
        logger.info("Full checkpoint written → %s", path)

    def load_full(self, path: str | Path) -> int:
        """Restore from a full checkpoint.  Returns the resume episode index."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        with torch.no_grad():
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
        ep = int(ckpt.get("episode", 0))
        logger.info("Resumed from %s @ episode %d", path, ep)
        return ep


# =====================================================================================
# ACTOR EXPORT (NUMPY) — BRAIN.PY COMPATIBLE
# =====================================================================================

def export_actor_npz(actor: SACActor, out_path: str | Path) -> None:
    """
    Serialize the actor's mean-head MLP as a NumPy ``.npz`` for live inference.

    The export contains the trunk layers followed by the deterministic
    ``mu`` head.  Both the new (``w0/b0/.../w3/b3``) and legacy
    (``W1/b1/W2/b2/W3/b3``) key conventions are written so the file is
    loadable by either generation of the live inference loop.

    Notes
    -----
    PyTorch stores ``Linear`` weights as ``(out_features, in_features)``.
    NumPy inference uses ``state @ W + b`` so we transpose to
    ``(in_features, out_features)`` on export.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pull the trunk's Linear layers (skip the activation modules).
    linear_modules: list[nn.Linear] = [
        m for m in actor.trunk.modules() if isinstance(m, nn.Linear)
    ]
    layers: list[nn.Linear] = list(linear_modules) + [actor.mu_head]

    payload: dict[str, np.ndarray] = {}
    for i, lin in enumerate(layers):
        w = lin.weight.detach().cpu().numpy().T.astype(np.float32)
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        payload[f"w{i}"] = w
        payload[f"b{i}"] = b

    # Legacy 3-layer compatibility (only valid when the network is exactly 3 layers).
    if len(layers) == 3:
        payload["W1"] = payload["w0"]
        payload["b1_legacy"] = payload["b0"]
        payload["W2"] = payload["w1"]
        payload["b2_legacy"] = payload["b1"]
        payload["W3"] = payload["w2"]
        payload["b3_legacy"] = payload["b2"]

    payload["trained"] = np.array([True])
    payload["state_dim"] = np.array([int(actor.state_dim)])
    payload["action_dim"] = np.array([int(actor.action_dim)])
    payload["hidden_dims"] = np.array(list(actor.hidden_dims), dtype=np.int64)
    payload["activation"] = np.array(["relu"])
    payload["squash"] = np.array(["tanh"])

    np.savez(str(out), **payload)
    logger.info(
        "Actor exported → %s (layers=%d, hidden=%s)",
        out, len(layers), actor.hidden_dims,
    )


# =====================================================================================
# CSV TRAINING LOG
# =====================================================================================

class TrainingLogger:
    """Append-only CSV logger for per-episode metrics."""

    FIELDS = [
        "episode", "phase", "total_reward", "net_pnl", "sharpe", "sortino",
        "calmar", "max_dd", "win_rate", "total_trades", "avg_trade_duration",
        "alpha", "actor_loss", "critic_loss", "regime_TRENDING",
        "regime_RANGING", "regime_HIGH_VOL", "curriculum_scale",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        new = not self.path.exists()
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in self.FIELDS}
        self._writer.writerow(out)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass


# =====================================================================================
# EPISODE SUMMARY PRINTER
# =====================================================================================

def print_episode_summary(
    episode: int,
    phase: str,
    total_reward: float,
    metrics: dict[str, float],
    regime_counts: dict[str, int],
    alpha: float,
    actor_loss: float,
    critic_loss: float,
    curriculum_scale: float,
) -> None:
    """Pretty-print a structured per-episode summary table to stdout."""
    line = "─" * 78
    title = f" Episode {episode:4d} — {phase} "
    pad = (78 - len(title)) // 2
    header = "─" * pad + title + "─" * (78 - pad - len(title))
    print(header)

    rows = [
        ("total_reward",        f"{total_reward:+.4f}"),
        ("net_pnl",             f"{metrics.get('net_pnl', 0.0) * 100:+.4f}%"),
        ("sharpe",              f"{metrics.get('sharpe', 0.0):+.3f}"),
        ("sortino",             f"{metrics.get('sortino', 0.0):+.3f}"),
        ("calmar",              f"{metrics.get('calmar', 0.0):+.3f}"),
        ("max_dd",              f"{metrics.get('max_dd', 0.0) * 100:.3f}%"),
        ("win_rate",            f"{metrics.get('win_rate', 0.0) * 100:.2f}%"),
        ("total_trades",        f"{int(metrics.get('total_trades', 0))}"),
        ("avg_trade_duration",  f"{metrics.get('avg_trade_duration', 0.0):.1f}"),
        ("alpha",               f"{alpha:.4f}"),
        ("actor_loss",          f"{actor_loss:+.4f}"),
        ("critic_loss",         f"{critic_loss:+.4f}"),
        ("curriculum_scale",    f"{curriculum_scale:.2f}"),
    ]
    for k, v in rows:
        print(f"  {k:<22}{v}")

    if regime_counts:
        total = max(1, sum(regime_counts.values()))
        regime_str = "  ".join(
            f"{k}={v} ({v / total * 100:.1f}%)" for k, v in regime_counts.items()
        )
        print(f"  regimes               {regime_str}")
    print(line)


# =====================================================================================
# TRAINING ORCHESTRATION
# =====================================================================================

class Trainer:
    """
    End-to-end training orchestrator: wraps env + agent + buffer + logging.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        device: torch.device,
        config_train: dict[str, Any],
        config_sac: dict[str, Any],
        config_reward: dict[str, Any],
    ) -> None:
        self.cfg_train = config_train
        self.cfg_sac = config_sac
        self.cfg_reward = config_reward
        self.device = device

        # Train / val split --------------------------------------------------
        n = len(df)
        val_n = int(n * float(config_train["val_split"]))
        self.train_end_idx = max(200, n - val_n)
        logger.info(
            "Train/val split: train=[0,%d)  val=[%d,%d)  total=%d",
            self.train_end_idx, self.train_end_idx, n, n,
        )

        # Scaler is fit ONLY on the training feature slice ------------------
        feats = list(DATA_CONFIG["feature_columns"])
        self.scaler = RunningStandardScaler().fit(
            df.loc[: self.train_end_idx - 1, feats].values.astype(np.float32)
        )

        # Reward + regime + env ---------------------------------------------
        self.reward_calc = RewardCalculator(self.cfg_reward)
        self.regime_detector = RegimeDetector()
        self.env_rng = np.random.default_rng(int(config_train["random_seed"]))
        self.env = CryptoTradingEnv(
            df=df,
            scaler=self.scaler,
            reward_calc=self.reward_calc,
            regime_detector=self.regime_detector,
            train_end_idx=self.train_end_idx,
            rng=self.env_rng,
            config=self.cfg_train,
        )

        # Agent / buffer ----------------------------------------------------
        hidden_dims, batch_size = device_dependent_config(device)
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        logger.info(
            "Network config: hidden_dims=%s  batch_size=%d  buffer=%d",
            hidden_dims, batch_size, int(self.cfg_sac["buffer_size"]),
        )

        self.agent = SACAgent(
            state_dim=int(self.cfg_sac["state_dim"]),
            action_dim=int(self.cfg_sac["action_dim"]),
            hidden_dims=hidden_dims,
            device=device,
            config=self.cfg_sac,
        )
        self.buffer = PrioritizedReplayBuffer(
            capacity=int(self.cfg_sac["buffer_size"]),
            state_dim=int(self.cfg_sac["state_dim"]),
            action_dim=int(self.cfg_sac["action_dim"]),
            alpha=float(self.cfg_sac["per_alpha"]),
            eps=float(self.cfg_sac["per_eps"]),
        )

        # Logging ------------------------------------------------------------
        self.csv_logger = TrainingLogger(self.cfg_train["log_path"])

        # Tracking ----------------------------------------------------------
        self.last_train_sharpe: float = 0.0
        self.start_episode: int = 0
        self._sigint_flag: bool = False
        self._step_counter: int = 0
        self._install_sigint_handler()

    # ── SIGINT ───────────────────────────────────────────────────────────────────────
    def _install_sigint_handler(self) -> None:
        def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            if self._sigint_flag:
                logger.error("Second SIGINT received — exiting hard.")
                sys.exit(1)
            self._sigint_flag = True
            logger.warning(
                "SIGINT received — finishing current step, will checkpoint & exit."
            )

        try:
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            # Not running on the main thread — silently skip.
            pass

    # ── PER beta annealing ───────────────────────────────────────────────────────────
    def _per_beta(self, episode: int) -> float:
        total = max(1, int(self.cfg_train["total_episodes"]))
        progress = min(1.0, episode / total)
        b0 = float(self.cfg_sac["per_beta_start"])
        b1 = float(self.cfg_sac["per_beta_end"])
        return b0 + (b1 - b0) * progress

    # ── Cosine LR schedule with warmup ──────────────────────────────────────────────
    def _lr_scale(self, episode: int) -> float:
        warmup = int(self.cfg_sac["warmup_episodes"])
        warm_scale = float(self.cfg_sac["warmup_lr_scale"])
        total = max(1, int(self.cfg_train["total_episodes"]))
        if episode < warmup:
            return warm_scale
        progress = (episode - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return warm_scale + (1.0 - warm_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ── Episode runner ───────────────────────────────────────────────────────────────
    def _run_episode(
        self,
        episode_idx: int,
        validation: bool,
    ) -> tuple[float, dict[str, float], dict[str, int]]:
        state = self.env.reset(episode_idx=episode_idx, validation=validation)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            if self._sigint_flag:
                break
            action = self.agent.act(state, deterministic=validation)
            next_state, reward, done, info = self.env.step(action)
            total_reward += float(reward)

            if not validation:
                self.buffer.push(
                    state=state,
                    action=np.array([action], dtype=np.float32),
                    reward=float(reward),
                    next_state=next_state,
                    done=done,
                )
                self._step_counter += 1

                ready = (
                    len(self.buffer) >= int(self.cfg_sac["min_buffer_to_train"])
                    and self._step_counter % int(self.cfg_sac["update_every"]) == 0
                )
                if ready:
                    beta = self._per_beta(episode_idx)
                    _, _, _, idxs, td_errors = self.agent.update(
                        buffer=self.buffer,
                        batch_size=self.batch_size,
                        beta=beta,
                    )
                    self.buffer.update_priorities(idxs, td_errors)

            state = next_state
            steps += 1

        metrics = self.env.episode_metrics()
        regimes = dict(self.reward_calc.regime_counts)
        return total_reward, metrics, regimes

    # ── Public train / validate ─────────────────────────────────────────────────────
    def train(self) -> None:
        """Main training loop — episodes drawn from the training slice."""
        total_eps = int(self.cfg_train["total_episodes"])
        logger.info("Starting SAC training for %d episodes", total_eps)
        for episode in range(self.start_episode, total_eps):
            if self._sigint_flag:
                break
            self.agent.set_lr_scale(self._lr_scale(episode))
            curric = self.env._curriculum_scale()
            total_reward, metrics, regimes = self._run_episode(
                episode_idx=episode, validation=False
            )

            self.last_train_sharpe = metrics.get("sharpe", 0.0)
            print_episode_summary(
                episode=episode,
                phase="TRAIN",
                total_reward=total_reward,
                metrics=metrics,
                regime_counts=regimes,
                alpha=self.agent.last_alpha,
                actor_loss=self.agent.last_actor_loss,
                critic_loss=self.agent.last_critic_loss,
                curriculum_scale=curric,
            )
            self.csv_logger.log({
                "episode": episode,
                "phase": "TRAIN",
                "total_reward": total_reward,
                "net_pnl": metrics.get("net_pnl", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "sortino": metrics.get("sortino", 0.0),
                "calmar": metrics.get("calmar", 0.0),
                "max_dd": metrics.get("max_dd", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "total_trades": metrics.get("total_trades", 0.0),
                "avg_trade_duration": metrics.get("avg_trade_duration", 0.0),
                "alpha": self.agent.last_alpha,
                "actor_loss": self.agent.last_actor_loss,
                "critic_loss": self.agent.last_critic_loss,
                "regime_TRENDING": regimes.get(RegimeDetector.REGIME_TRENDING, 0),
                "regime_RANGING": regimes.get(RegimeDetector.REGIME_RANGING, 0),
                "regime_HIGH_VOL": regimes.get(RegimeDetector.REGIME_HIGH_VOL, 0),
                "curriculum_scale": curric,
            })

            # Periodic checkpoint -----------------------------------------------
            if (episode + 1) % int(self.cfg_train["checkpoint_every"]) == 0:
                self._write_checkpoints(episode + 1)

            # Periodic validation ----------------------------------------------
            if (
                (episode + 1) % int(self.cfg_train["validate_every"]) == 0
                and (episode + 1) >= int(self.cfg_train["validate_every"])
            ):
                self._run_validation(episode_idx=episode)

        # Final flush ------------------------------------------------------------
        self._write_checkpoints(self.cfg_train["total_episodes"])
        self.csv_logger.close()
        if self._sigint_flag:
            logger.warning("Training stopped early by SIGINT.")
        else:
            logger.info("Training complete.")

    def _run_validation(self, episode_idx: int) -> dict[str, float]:
        """Run a single deterministic validation episode on held-out data."""
        logger.info("Running out-of-sample validation episode...")
        total_reward, metrics, regimes = self._run_episode(
            episode_idx=episode_idx, validation=True
        )
        print_episode_summary(
            episode=episode_idx,
            phase="VAL_",
            total_reward=total_reward,
            metrics=metrics,
            regime_counts=regimes,
            alpha=self.agent.last_alpha,
            actor_loss=self.agent.last_actor_loss,
            critic_loss=self.agent.last_critic_loss,
            curriculum_scale=1.0,
        )
        self.csv_logger.log({
            "episode": episode_idx,
            "phase": "VAL_",
            "total_reward": total_reward,
            "net_pnl": metrics.get("net_pnl", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "sortino": metrics.get("sortino", 0.0),
            "calmar": metrics.get("calmar", 0.0),
            "max_dd": metrics.get("max_dd", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_trades": metrics.get("total_trades", 0.0),
            "avg_trade_duration": metrics.get("avg_trade_duration", 0.0),
            "alpha": self.agent.last_alpha,
            "actor_loss": self.agent.last_actor_loss,
            "critic_loss": self.agent.last_critic_loss,
            "regime_TRENDING": regimes.get(RegimeDetector.REGIME_TRENDING, 0),
            "regime_RANGING": regimes.get(RegimeDetector.REGIME_RANGING, 0),
            "regime_HIGH_VOL": regimes.get(RegimeDetector.REGIME_HIGH_VOL, 0),
            "curriculum_scale": 1.0,
        })
        # Over-fitting watchdog ------------------------------------------------
        delta = self.last_train_sharpe - metrics.get("sharpe", 0.0)
        if delta > float(self.cfg_train["validation_sharpe_alert_delta"]):
            logger.warning(
                "Overfitting alert — train Sharpe %.3f exceeds val Sharpe %.3f by %.3f",
                self.last_train_sharpe, metrics.get("sharpe", 0.0), delta,
            )
        return metrics

    # ── Checkpoint helpers ──────────────────────────────────────────────────────────
    def _write_checkpoints(self, episode: int) -> None:
        export_actor_npz(self.agent.actor, self.cfg_train["actor_export_path"])
        self.agent.save_full(self.cfg_train["checkpoint_path"], episode=episode)

    def resume(self) -> None:
        """Restore from ``sac_full.pt`` if present."""
        path = Path(self.cfg_train["checkpoint_path"])
        if not path.exists():
            logger.warning("No checkpoint found at %s — starting fresh.", path)
            return
        ep = self.agent.load_full(path)
        self.start_episode = ep

    def validate_only(self) -> None:
        """Run a single validation episode and exit (no training)."""
        path = Path(self.cfg_train["checkpoint_path"])
        if path.exists():
            self.agent.load_full(path)
        else:
            logger.warning(
                "No %s found — running validation against freshly-initialised actor.",
                path,
            )
        self._run_validation(episode_idx=0)


# =====================================================================================
# CLI
# =====================================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse interface."""
    p = argparse.ArgumentParser(
        description="SAC offline trainer with institutional-grade reward.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  python offline_trainer.py
  python offline_trainer.py --episodes 500
  python offline_trainer.py --resume
  python offline_trainer.py --validate-only
  python offline_trainer.py --device cpu --data my.csv
""",
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="CSV path override (defaults to DATA_CONFIG['csv_path']).",
    )
    p.add_argument(
        "--episodes", type=int, default=None,
        help="Override TRAIN_CONFIG['total_episodes'].",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Load sac_full.pt and continue training from the saved episode.",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Run a single deterministic validation episode and exit.",
    )
    p.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Force device (auto-detected when omitted).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override TRAIN_CONFIG['random_seed'].",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        help="Python logging level (DEBUG / INFO / WARNING / ERROR).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point — wire argparse → Trainer."""
    args = _parse_args(argv)

    # Logging level ----------------------------------------------------------
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    # Apply CLI overrides ---------------------------------------------------
    if args.data is not None:
        TRAIN_CONFIG["data_path"] = args.data
        DATA_CONFIG["csv_path"] = args.data
    if args.episodes is not None:
        TRAIN_CONFIG["total_episodes"] = int(args.episodes)
    if args.seed is not None:
        TRAIN_CONFIG["random_seed"] = int(args.seed)
    if args.device is not None:
        TRAIN_CONFIG["device_override"] = args.device

    # Seeds & device --------------------------------------------------------
    set_global_seeds(int(TRAIN_CONFIG["random_seed"]))
    device = resolve_device(TRAIN_CONFIG["device_override"])
    log_memory_budget(device)

    # Dataset ----------------------------------------------------------------
    df = load_dataset(TRAIN_CONFIG["data_path"])
    logger.info("Dataset: rows=%d  columns=%d", len(df), df.shape[1])

    # Trainer ----------------------------------------------------------------
    trainer = Trainer(
        df=df,
        device=device,
        config_train=TRAIN_CONFIG,
        config_sac=SAC_CONFIG,
        config_reward=REWARD_CONFIG,
    )

    if args.resume:
        trainer.resume()

    if args.validate_only:
        trainer.validate_only()
        return 0

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user — flushing checkpoint.")
        trainer._write_checkpoints(episode=int(TRAIN_CONFIG["total_episodes"]))
        trainer.csv_logger.close()
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
