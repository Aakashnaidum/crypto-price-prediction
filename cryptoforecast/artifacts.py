"""Save/load deployable model bundles and make single-candle predictions.

A bundle stores the fitted estimator *together with* the feature list, target
definition and evaluation evidence, so the web app cannot drift out of sync
with how the model was trained.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import MODELS_DIR
from .features import FEATURE_SETS, candle_features

BUNDLE_VERSION = 2


def bundle_path(asset_key: str, models_dir: Path | None = None) -> Path:
    return Path(models_dir or MODELS_DIR) / f"{asset_key}.joblib"


def save_bundle(path: Path, *, asset_key, model, trained_through, evidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "bundle_version": BUNDLE_VERSION,
            "asset": asset_key,
            "model_label": model.label,
            "feature_set": model.feature_set,
            "features": FEATURE_SETS[model.feature_set],
            "target": "log(next_close / close)",
            "estimator": model.model_,
            "trained_through": trained_through,
            "sklearn_version": sklearn.__version__,
            "evidence": evidence,
        },
        path,
    )


def load_bundle(path: Path) -> dict:
    """Load a bundle. Only load files you created yourself: joblib uses pickle."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"{path} is not a v{BUNDLE_VERSION} model bundle; re-run scripts/train_models.py")
    if bundle["feature_set"] != "candle":
        raise ValueError("The web app can only serve single-candle ('candle') models")
    return bundle


def validate_candle(open_, high, low, close, volume) -> None:
    values = {"Open": open_, "High": high, "Low": low, "Close": close}
    for k, v in values.items():
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{k} must be a positive number")
    if not np.isfinite(volume) or volume < 0:
        raise ValueError("Volume must be a non-negative number")
    if low > min(open_, close) or high < max(open_, close) or low > high:
        raise ValueError("Prices must satisfy Low <= min(Open, Close) and High >= max(Open, Close)")


def predict_next_close(bundle: dict, *, open_, high, low, close, volume) -> dict:
    """Forecast the next daily close from one completed candle."""
    validate_candle(open_, high, low, close, volume)
    feats = candle_features(open_, high, low, close, volume)
    row = pd.DataFrame([{k: float(feats[k]) for k in bundle["features"]}])
    log_ret = float(bundle["estimator"].predict(row)[0])
    return {
        "model_forecast": close * float(np.exp(log_ret)),
        "persistence_forecast": close,
        "predicted_log_return": log_ret,
        "predicted_change_pct": (np.exp(log_ret) - 1) * 100,
    }
