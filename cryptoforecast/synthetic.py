"""Synthetic OHLCV generator used by tests and CI (no real market data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_history(n_days: int = 400, seed: int = 0, start: str = "2020-01-01", start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0005, 0.03, n_days)
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(rng.normal(0, 0.002, n_days))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n_days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n_days)))
    volume = rng.lognormal(15, 0.5, n_days)
    return pd.DataFrame({
        "Date": pd.date_range(start, periods=n_days, freq="D"),
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    })
