"""Forecasters with a common interface: fit on a dataset, predict next close price.

Every forecaster exposes ``fit(frame)`` and ``predict_price(frame)`` where
``frame`` is a slice of :func:`cryptoforecast.features.build_dataset` output.
Return-space models predict ``log(next_close / close)`` and convert back to a
price, which lets tree models work across price regimes they never saw during
training (a level-space tree cannot predict above its training maximum).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE
from .features import FEATURE_SETS


class Forecaster:
    name: str = "base"
    feature_set: str | None = None
    learned: bool = True

    def fit(self, frame: pd.DataFrame) -> "Forecaster":
        raise NotImplementedError

    def predict_price(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return f"{self.name}[{self.feature_set}]" if self.feature_set else self.name


class Persistence(Forecaster):
    """Random-walk baseline: tomorrow's close = today's close."""

    name = "Persistence"
    learned = False

    def fit(self, frame):
        return self

    def predict_price(self, frame):
        return frame["Close"].to_numpy(dtype=float)


class Drift(Forecaster):
    """Random walk with drift: today's close times exp(mean training log return)."""

    name = "Drift"
    learned = False

    def fit(self, frame):
        self.mu_ = float(frame["target_log_return"].mean())
        return self

    def predict_price(self, frame):
        return frame["Close"].to_numpy(dtype=float) * np.exp(self.mu_)


class LevelModel(Forecaster):
    """Original project formulation: regress next close *level* on raw inputs."""

    def __init__(self, name: str, estimator, feature_set: str = "legacy_level"):
        self.name, self.estimator, self.feature_set = name, estimator, feature_set

    def fit(self, frame):
        self.model_ = clone(self.estimator).fit(frame[FEATURE_SETS[self.feature_set]], frame["next_close"])
        return self

    def predict_price(self, frame):
        return self.model_.predict(frame[FEATURE_SETS[self.feature_set]])


class ReturnModel(Forecaster):
    """Predict next-day log return, then convert to a price forecast."""

    def __init__(self, name: str, estimator, feature_set: str = "candle"):
        self.name, self.estimator, self.feature_set = name, estimator, feature_set

    def fit(self, frame):
        self.model_ = clone(self.estimator).fit(frame[FEATURE_SETS[self.feature_set]], frame["target_log_return"])
        return self

    def predict_log_return(self, frame) -> np.ndarray:
        return self.model_.predict(frame[FEATURE_SETS[self.feature_set]])

    def predict_price(self, frame):
        return frame["Close"].to_numpy(dtype=float) * np.exp(self.predict_log_return(frame))


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _rf():
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, max_features=0.5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def candidate_forecasters() -> list[Forecaster]:
    """All forecasters compared in the study (fresh, unfitted instances)."""
    models: list[Forecaster] = [Persistence(), Drift()]
    # Reproduction of the earlier project setup (level target, raw inputs).
    models += [
        LevelModel("LinearRegression", LinearRegression()),
        LevelModel("DecisionTree", DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE)),
        LevelModel("RandomForest", RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    for fs in ("candle", "candle_lags"):
        models += [
            ReturnModel("Ridge", _ridge(), fs),
            ReturnModel("RandomForest", _rf(), fs),
            ReturnModel("HistGradientBoosting", _hgb(), fs),
        ]
    return models


def deployable_forecasters() -> list[ReturnModel]:
    """Models the web app can serve: they need only one completed daily candle."""
    return [m for m in candidate_forecasters() if isinstance(m, ReturnModel) and m.feature_set == "candle"]
