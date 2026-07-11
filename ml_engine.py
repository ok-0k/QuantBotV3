"""
ml_engine.py — XGBoost incremental learning engine.

Designed to run ENTIRELY inside ProcessPoolExecutor workers —
never on the asyncio event loop thread.

Key design choices:
  - Incremental (online) learning via xgb_model continuation parameter
  - Binary classification: predict P(profitable trade) not price
  - DMatrix format for maximum memory efficiency on ARM64
  - nthread=2 to prevent thermal throttling on Pi 5
  - Model serialised to disk after every update cycle
"""

from __future__ import annotations
import fcntl
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logging.warning("XGBoost not installed — ML engine running in STUB mode")

from config import (XGB_MODEL, XGB_PARAMS, XGB_INCREMENTAL_TREES,
                    ML_SIGNAL_THRESHOLD, ML_MIN_SAMPLES)
from features import compute_features, compute_targets, get_live_feature_vector

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SINGLETON  (within a worker process)
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional["xgb.XGBClassifier"] = None


def _load_model() -> Optional["xgb.XGBClassifier"]:
    """Load model from disk if it exists, otherwise return None (untrained)."""
    global _model
    if not _HAS_XGB:
        return None
    if _model is not None:
        return _model
    if XGB_MODEL.exists():
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.load_model(str(XGB_MODEL))
        _model = m
        log.info("XGBoost model loaded from %s", XGB_MODEL)
    return _model


def _save_model(model: "xgb.XGBClassifier") -> None:
    global _model
    model.save_model(str(XGB_MODEL))
    _model = model
    log.info("XGBoost model saved → %s", XGB_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (called via run_in_executor)
# ─────────────────────────────────────────────────────────────────────────────

def predict_signal(candles: list[dict]) -> float:
    """
    Run XGBoost inference on the latest candle's feature vector.

    Returns probability ∈ [0, 1] that the next N candles yield profit.
    Returns 0.5 (neutral) when model is untrained or feature data insufficient.

    This function blocks during CPU-bound inference but is executed in a
    worker process, keeping the asyncio event loop non-blocking.
    """
    if not _HAS_XGB:
        return 0.5

    model = _load_model()
    if model is None:
        return 0.5   # untrained — neutral signal

    fvec = get_live_feature_vector(candles)
    if fvec is None:
        return 0.5

    try:
        prob = float(model.predict_proba(fvec.reshape(1, -1))[0, 1])
        return prob
    except Exception as exc:
        log.error("XGBoost predict failed: %s", exc)
        return 0.5


def incremental_train(candles: list[dict]) -> bool:
    """
    Incrementally update the XGBoost model with new data.

    Uses gradient boosting's additive nature: existing trees are preserved,
    new trees are added to correct residual errors on recent data.

    Scheduled as an async cron task — runs in worker process to avoid
    blocking the event loop or causing thermal issues on the Pi 5.

    Returns True if training succeeded.
    """
    if not _HAS_XGB:
        return False

    feat_df, names = compute_features(candles)
    if feat_df.empty or len(feat_df) < ML_MIN_SAMPLES:
        log.info("Incremental train skipped — only %d samples (need %d)",
                 len(feat_df), ML_MIN_SAMPLES)
        return False

    targets = compute_targets(candles)

    # Align targets with feature rows (features have NaN rows dropped)
    # targets is indexed on original candle index; feat_df.index tells us which survived
    valid_idx = feat_df.index
    if valid_idx[-1] >= len(targets):
        valid_idx = valid_idx[valid_idx < len(targets)]
    if len(valid_idx) < ML_MIN_SAMPLES:
        return False

    X = feat_df.loc[valid_idx].values.astype(np.float32)
    y = targets.iloc[valid_idx].values.astype(int)

    # Skip last TARGET_FORWARD_CANDLES rows — targets not yet observable
    from config import TARGET_FORWARD_CANDLES
    X = X[:-TARGET_FORWARD_CANDLES]
    y = y[:-TARGET_FORWARD_CANDLES]

    if len(X) < ML_MIN_SAMPLES:
        return False

    current_model = _load_model()

    try:
        new_model = xgb.XGBClassifier(
            n_estimators=XGB_INCREMENTAL_TREES,
            **{k: v for k, v in XGB_PARAMS.items() if k != "n_estimators"},
        )

        if current_model is not None:
            # ── INCREMENTAL LEARNING ─────────────────────────────────────────
            # Preserves existing trees; adds XGB_INCREMENTAL_TREES new trees
            # that correct residual errors on the newly provided data.
            # This is the core mechanism that allows continuous self-improvement
            # without full retraining (which would thermally stress the Pi 5).
            new_model.fit(X, y, xgb_model=current_model)
        else:
            # First train — no prior model
            new_model = xgb.XGBClassifier(**XGB_PARAMS)
            new_model.fit(X, y)

        _save_model(new_model)
        log.info("Incremental XGBoost update — %d samples, class balance %.2f%%",
                 len(X), y.mean() * 100)
        return True

    except Exception as exc:
        log.error("Incremental training failed: %s", exc)
        return False


def initial_train(candles_per_symbol: dict[str, list[dict]]) -> bool:
    """
    Full initial training across all symbols' candle history.
    Called once on startup if no model exists.
    """
    if not _HAS_XGB or XGB_MODEL.exists():
        return False

    all_X, all_y = [], []
    for sym, candles in candles_per_symbol.items():
        feat_df, _ = compute_features(candles)
        if feat_df.empty:
            continue
        targets = compute_targets(candles)

        valid_idx = feat_df.index
        if valid_idx[-1] >= len(targets):
            valid_idx = valid_idx[valid_idx < len(targets)]

        from config import TARGET_FORWARD_CANDLES
        X = feat_df.loc[valid_idx].values.astype(np.float32)[:-TARGET_FORWARD_CANDLES]
        y = targets.iloc[valid_idx].values.astype(int)[:-TARGET_FORWARD_CANDLES]

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        log.warning("Initial training: no usable data across all symbols")
        return False

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    log.info("Initial XGBoost training — %d samples from %d symbols",
             len(X_all), len(all_X))

    try:
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_all, y_all)
        _save_model(model)
        log.info("Initial XGBoost model trained and saved")
        return True
    except Exception as exc:
        log.error("Initial training failed: %s", exc)
        return False


def get_feature_importance() -> dict[str, float]:
    """Return feature importance dict for dashboard display."""
    model = _load_model()
    if model is None or not hasattr(model, "feature_importances_"):
        return {}
    scores = model.feature_importances_
    # Feature names come from compute_features; we don't have them here
    return {f"f{i}": float(s) for i, s in enumerate(scores)}

