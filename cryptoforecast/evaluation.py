"""Chronological evaluation, metrics and a forecast-accuracy significance test."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pandas as pd

from .config import TEST_FRACTION, VALIDATION_FRACTION, WALK_FORWARD_FOLDS
from .features import FEATURE_SETS
from .models import Forecaster, Persistence


# ----------------------------------------------------------------- metrics --
def directional_accuracy(close_t, predicted_next, actual_next) -> float:
    """Share of days where the predicted move sign equals the realised sign.

    Days with no realised move and days where the model predicts exactly no
    move are excluded. Returns NaN for a forecaster that never predicts a move
    (e.g. persistence), because "direction" is undefined for it.
    """
    pred = np.sign(np.asarray(predicted_next) - np.asarray(close_t))
    true = np.sign(np.asarray(actual_next) - np.asarray(close_t))
    mask = (true != 0) & (pred != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == true[mask]).mean())


def point_metrics(close_t, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(np.mean(np.abs(err) / y_true) * 100),
        "directional_accuracy": directional_accuracy(close_t, y_pred, y_true),
        "n": int(len(y_true)),
    }


def diebold_mariano(e_model, e_base, horizon: int = 1) -> dict:
    """Diebold-Mariano test (squared-error loss) with Harvey et al. correction.

    Negative statistic => the model's squared errors are lower than the
    baseline's. p-value is two-sided using a Student-t with n-1 d.o.f.
    """
    from scipy import stats

    d = np.asarray(e_model, float) ** 2 - np.asarray(e_base, float) ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    d_mean = d.mean()
    gamma0 = np.var(d, ddof=0)
    acov = sum(
        2 * np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, horizon)
    ) if horizon > 1 else 0.0
    var = (gamma0 + acov) / n
    if var <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dm = d_mean / math.sqrt(var)
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm *= correction
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return {"dm_stat": float(dm), "p_value": float(p)}


# ------------------------------------------------------------------ splits --
def common_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """Rows usable by every feature set, so all models are scored on the same days."""
    all_cols = sorted({c for cols in FEATURE_SETS.values() for c in cols})
    return dataset.dropna(subset=all_cols).reset_index(drop=True)


def chronological_split(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    cut = int(len(frame) * (1 - test_fraction))
    return frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)


def walk_forward_splits(n: int, folds: int = WALK_FORWARD_FOLDS, start_fraction: float = 0.5):
    """Expanding-window folds over the second half of the series."""
    start = int(n * start_fraction)
    size = max(1, (n - start) // folds)
    for k in range(folds):
        tr_end = start + k * size
        te_end = n if k == folds - 1 else tr_end + size
        if te_end > tr_end:
            yield np.arange(0, tr_end), np.arange(tr_end, te_end)


# -------------------------------------------------------------- evaluation --
def evaluate_holdout(frame: pd.DataFrame, forecasters: list[Forecaster]) -> dict:
    train, test = chronological_split(frame)
    base = Persistence().fit(train).predict_price(test)
    y = test["next_close"].to_numpy()
    close_t = test["Close"].to_numpy()
    base_err = base - y
    base_mae = np.mean(np.abs(base_err))

    results = {}
    for f in forecasters:
        model = deepcopy(f).fit(train)
        pred = model.predict_price(test)
        m = point_metrics(close_t, y, pred)
        m["mae_vs_persistence"] = float(m["mae"] / base_mae)
        m.update(diebold_mariano(pred - y, base_err) if f.name != "Persistence" else
                 {"dm_stat": float("nan"), "p_value": float("nan")})
        results[f.label] = m
    return {
        "train_start": str(train["Date"].iloc[0].date()),
        "train_end": str(train["Date"].iloc[-1].date()),
        "test_start": str(test["Date"].iloc[0].date()),
        "test_end": str(test["Date"].iloc[-1].date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "models": results,
    }


def evaluate_walk_forward(frame: pd.DataFrame, forecasters: list[Forecaster]) -> dict:
    out = {}
    for f in forecasters:
        rmses, rel_maes = [], []
        for tr_idx, te_idx in walk_forward_splits(len(frame)):
            tr, te = frame.iloc[tr_idx], frame.iloc[te_idx]
            y = te["next_close"].to_numpy()
            pred = deepcopy(f).fit(tr).predict_price(te)
            base = te["Close"].to_numpy()
            rmses.append(float(np.sqrt(np.mean((pred - y) ** 2))))
            rel_maes.append(float(np.mean(np.abs(pred - y)) / np.mean(np.abs(base - y))))
        out[f.label] = {
            "folds": len(rmses),
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_mae_vs_persistence": float(np.mean(rel_maes)),
            "folds_beating_persistence": int(sum(r < 1 for r in rel_maes)),
        }
    return out


def select_on_validation(train: pd.DataFrame, forecasters: list[Forecaster]) -> tuple[str, dict]:
    """Pick a model using only the training block (inner chronological split)."""
    inner_train, valid = chronological_split(train, VALIDATION_FRACTION)
    y = valid["next_close"].to_numpy()
    scores = {}
    for f in forecasters:
        pred = deepcopy(f).fit(inner_train).predict_price(valid)
        scores[f.label] = float(np.mean(np.abs(pred - y)))
    return min(scores, key=scores.get), scores
