import numpy as np
import pandas as pd
import pytest

from cryptoforecast.data import DataValidationError, data_quality_report, load_history
from cryptoforecast.features import FEATURE_SETS, build_dataset
from cryptoforecast.synthetic import synthetic_history


def test_target_is_next_calendar_day_close():
    hist = synthetic_history(60)
    ds = build_dataset(hist)
    by_date = hist.set_index("Date")["Close"]
    for _, row in ds.sample(10, random_state=0).iterrows():
        assert row["next_date"] == row["Date"] + pd.Timedelta(days=1)
        assert row["next_close"] == pytest.approx(by_date[row["next_date"]])
        assert row["target_log_return"] == pytest.approx(np.log(row["next_close"] / row["Close"]))


def test_features_do_not_use_future_values():
    """Changing prices after day t must not change any feature on day t."""
    hist = synthetic_history(120)
    t = 80
    altered = hist.copy()
    altered.loc[t + 1:, ["Open", "High", "Low", "Close", "Volume"]] *= 3.0
    cols = sorted({c for cs in FEATURE_SETS.values() for c in cs})
    a = build_dataset(hist).set_index("Date").loc[: hist["Date"][t], cols]
    b = build_dataset(altered).set_index("Date").loc[: hist["Date"][t], cols]
    pd.testing.assert_frame_equal(a, b)


def test_gap_days_are_not_paired_or_used_as_one_day_returns():
    hist = synthetic_history(60).drop(index=[30, 31])  # two missing calendar days
    ds = build_dataset(hist)
    assert (ds["next_date"] - ds["Date"] == pd.Timedelta(days=1)).all()
    assert hist["Date"][29] not in set(ds["Date"])  # its next day is missing
    after_gap = ds.set_index("Date").loc[hist["Date"][32]]
    assert np.isnan(after_gap["ret_0"])  # return across the gap is undefined


def test_load_history_validates(tmp_path):
    hist = synthetic_history(10)
    good = tmp_path / "good.csv"
    hist.to_csv(good, index=False)
    assert len(load_history(good)) == 10
    bad = hist.copy()
    bad.loc[3, "Close"] = -1
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        load_history(p)
    with pytest.raises(DataValidationError):
        hist.drop(columns="Volume").to_csv(p, index=False)
        load_history(p)


def test_quality_report_counts_gaps():
    hist = synthetic_history(20).drop(index=[5, 6, 12]).reset_index(drop=True)
    rep = data_quality_report(hist)
    assert rep["missing_calendar_days"] == 3
    assert rep["gap_events"] == 2
