"""Loading and validating daily OHLCV price histories."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ASSETS, DATA_DIR

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED_COLUMNS = ["Date", *PRICE_COLUMNS]


class DataValidationError(ValueError):
    """Raised when a price file cannot be used safely."""


def csv_path_for(asset_key: str, data_dir: Path | None = None) -> Path:
    if asset_key not in ASSETS:
        raise KeyError(f"Unknown asset '{asset_key}'. Known: {', '.join(ASSETS)}")
    return Path(data_dir or DATA_DIR) / ASSETS[asset_key].csv_name


def load_history(path: Path | str) -> pd.DataFrame:
    """Load a daily OHLCV CSV, normalise dates to UTC calendar days and validate it.

    Returns one row per calendar day that is present in the file, sorted by date.
    Missing days are *not* filled here; feature construction handles gaps
    explicitly so a multi-day jump is never mistaken for a one-day return.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found: {path}. See data/README.md for how to obtain the data."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name} is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    dates = pd.to_datetime(df["Date"], format="mixed", utc=True)
    df["Date"] = dates.dt.tz_convert(None).dt.normalize()
    df = df.dropna(subset=PRICE_COLUMNS)
    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise DataValidationError(f"{path.name} contains non-positive prices")
    if (df["Volume"] < 0).any():
        raise DataValidationError(f"{path.name} contains negative volume")

    df = df.sort_values("Date")
    dupes = df["Date"].duplicated(keep="last")
    df = df.loc[~dupes].reset_index(drop=True)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise coverage issues that matter for next-day forecasting."""
    gaps = df["Date"].diff().dt.days
    ohlc_violations = (
        (df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-12)
        | (df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-12)
    )
    return {
        "rows": int(len(df)),
        "first_date": str(df["Date"].min().date()),
        "last_date": str(df["Date"].max().date()),
        "missing_calendar_days": int((gaps[gaps > 1] - 1).sum()),
        "gap_events": int((gaps > 1).sum()),
        "zero_volume_rows": int((df["Volume"] == 0).sum()),
        "ohlc_inconsistent_rows": int(ohlc_violations.sum()),
    }
