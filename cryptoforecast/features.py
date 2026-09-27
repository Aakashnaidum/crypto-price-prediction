"""Feature construction for one-day-ahead close forecasting.

Timing convention (the core leakage guard)
------------------------------------------
Row ``t`` holds the *completed* daily candle for calendar day ``t``. Every
feature on row ``t`` uses only values from day ``t`` or earlier. The target on
row ``t`` is the close of calendar day ``t + 1``. Rows whose next calendar day
is missing from the data are dropped rather than silently pairing day ``t``
with a later day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature sets ---------------------------------------------------------------
# legacy_level: the original project's inputs (same-day High/Low/Open/Volume),
#   predicting the next close *level*. Kept only to reproduce the saved results.
LEGACY_FEATURES = ["High", "Low", "Open", "Volume"]

# candle: scale-free descriptors of a single completed candle. These are what
#   the web app can compute from one day's OHLCV entered by a user.
CANDLE_FEATURES = ["log_range", "log_body", "upper_wick", "lower_wick", "log_volume"]

# candle_lags: candle features plus short return/volatility history.
LAG_FEATURES = [
    "ret_0", "ret_1", "ret_2", "ret_3", "ret_4",
    "vol_7", "ma_gap_7", "ma_gap_30", "volume_change",
]

FEATURE_SETS = {
    "legacy_level": LEGACY_FEATURES,
    "candle": CANDLE_FEATURES,
    "candle_lags": CANDLE_FEATURES + LAG_FEATURES,
}


def candle_features(open_, high, low, close, volume) -> dict:
    """Scale-free features of one completed candle (works on scalars or arrays)."""
    open_, high, low, close, volume = (np.asarray(v, dtype=float) for v in (open_, high, low, close, volume))
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    return {
        "log_range": np.log(high / low),
        "log_body": np.log(close / open_),
        "upper_wick": np.log(high / body_top),
        "lower_wick": np.log(body_bottom / low),
        "log_volume": np.log1p(volume),
    }


def build_dataset(history: pd.DataFrame) -> pd.DataFrame:
    """Return a supervised dataset with all feature sets and targets.

    Columns added:
      next_date, next_close  -- the realised close of the following calendar day
      target_log_return      -- log(next_close / Close)
    """
    daily = history.set_index("Date").asfreq("D")  # missing days become NaN rows

    out = daily.copy()
    feats = candle_features(daily["Open"], daily["High"], daily["Low"], daily["Close"], daily["Volume"])
    for name, values in feats.items():
        out[name] = values

    log_close = np.log(daily["Close"])
    ret = log_close.diff()  # NaN across any gap, so no multi-day "one-day" returns
    for lag in range(5):
        out[f"ret_{lag}"] = ret.shift(lag)
    out["vol_7"] = ret.rolling(7, min_periods=7).std()
    out["ma_gap_7"] = log_close - np.log(daily["Close"].rolling(7, min_periods=7).mean())
    out["ma_gap_30"] = log_close - np.log(daily["Close"].rolling(30, min_periods=30).mean())
    out["volume_change"] = np.log1p(daily["Volume"]) - np.log1p(daily["Volume"].shift(1))

    out["next_close"] = daily["Close"].shift(-1)
    out["next_date"] = out.index + pd.Timedelta(days=1)
    out["target_log_return"] = np.log(out["next_close"] / out["Close"])

    out = out.dropna(subset=["Close", "next_close"])  # both days must exist
    out.index.name = "Date"
    return out.reset_index()


def usable_rows(dataset: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cols = FEATURE_SETS[feature_set]
    return dataset.dropna(subset=cols).reset_index(drop=True)
