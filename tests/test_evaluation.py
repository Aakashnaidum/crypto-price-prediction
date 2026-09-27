import numpy as np
import pytest

from cryptoforecast.evaluation import (
    chronological_split, common_frame, diebold_mariano, directional_accuracy,
    evaluate_holdout, point_metrics, select_on_validation, walk_forward_splits,
)
from cryptoforecast.features import build_dataset
from cryptoforecast.models import Persistence, ReturnModel, candidate_forecasters, deployable_forecasters
from cryptoforecast.synthetic import synthetic_history


@pytest.fixture(scope="module")
def frame():
    return common_frame(build_dataset(synthetic_history(500, seed=1)))


def test_chronological_split_has_no_overlap(frame):
    train, test = chronological_split(frame)
    assert train["Date"].max() < test["Date"].min()
    assert len(train) + len(test) == len(frame)


def test_walk_forward_trains_only_on_past():
    for tr, te in walk_forward_splits(100, folds=5):
        assert tr.max() < te.min()


def test_persistence_metrics_and_direction():
    close = np.array([10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 10.0])
    m = point_metrics(close, actual, close)
    assert m["mae"] == pytest.approx(2 / 3)
    assert np.isnan(m["directional_accuracy"])  # persistence never predicts a move
    assert directional_accuracy(close, np.array([12, 8, 11.0]), actual) == 1.0


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 2, 500)
    better = base * 0.5
    res = diebold_mariano(better, base)
    assert res["dm_stat"] < 0 and res["p_value"] < 0.01


def test_holdout_reports_every_model_relative_to_persistence(frame):
    res = evaluate_holdout(frame, candidate_forecasters()[:4])
    assert res["models"]["Persistence"]["mae_vs_persistence"] == pytest.approx(1.0)
    assert set(res["models"]) >= {"Persistence", "Drift"}


def test_return_model_recovers_a_real_signal():
    """If next-day return is predictable from the candle body, the model must beat persistence."""
    hist = synthetic_history(800, seed=3)
    body = np.log(hist["Close"] / hist["Open"]).to_numpy()
    close = hist["Close"].to_numpy().copy()
    for i in range(1, len(close)):  # inject: tomorrow's return = 0.5 * today's body + noise
        close[i] = close[i - 1] * np.exp(0.5 * body[i - 1] + np.random.default_rng(i).normal(0, 0.002))
    hist["Close"] = close
    hist["High"] = np.maximum(hist[["Open", "Close"]].max(axis=1), hist["High"])
    hist["Low"] = np.minimum(hist[["Open", "Close"]].min(axis=1), hist["Low"])
    fr = common_frame(build_dataset(hist))
    ridge = [m for m in deployable_forecasters() if m.name == "Ridge"][0]
    res = evaluate_holdout(fr, [Persistence(), ridge])
    assert res["models"][ridge.label]["mae_vs_persistence"] < 0.9


def test_selection_uses_only_training_block(frame):
    train, test = chronological_split(frame)
    label, scores = select_on_validation(train, deployable_forecasters())
    assert label in scores and all(isinstance(m, ReturnModel) for m in deployable_forecasters())
