"""Rebuild data/<Asset>.csv from the two public sources used in this project.

1. Kaggle "Cryptocurrency Prices Data" (maharshipandya), daily rows up to 2022-10-23.
   Download it yourself (Kaggle account required) and pass --kaggle-csv.
2. Yahoo Finance daily candles (via the `yfinance` package) for every day after
   the last Kaggle date. Pass --yahoo-dir to use previously saved CSVs instead
   of downloading (columns: Date, Open, High, Low, Close, Volume).

Note on the splice: the two sources are different aggregators, so there can be
small level differences at the boundary. The evaluation hold-out periods lie
entirely inside the Yahoo segment.

Usage:
  python scripts/build_dataset.py --kaggle-csv ~/Downloads/cryptocurrency_prices_data_kaggle.csv
  python scripts/build_dataset.py --kaggle-csv K.csv --yahoo-dir saved_yahoo/ --end 2026-05-10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from cryptoforecast.config import ASSETS, DATA_DIR  # noqa: E402

COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Source"]


def kaggle_rows(kaggle: pd.DataFrame, name: str) -> pd.DataFrame:
    k = kaggle[kaggle["crypto_name"] == name].rename(columns=str.capitalize)
    if k.empty:
        raise ValueError(f"'{name}' not found in the Kaggle file")
    k["Date"] = pd.to_datetime(k["Date"]).dt.normalize()
    k["Source"] = "kaggle"
    return k[COLUMNS]


def yahoo_rows(asset, start: pd.Timestamp, end: str | None, yahoo_dir: Path | None) -> pd.DataFrame:
    if yahoo_dir is not None:
        y = pd.read_csv(yahoo_dir / f"{asset.key}_daily_history.csv")
    else:
        import yfinance as yf  # optional dependency

        y = yf.download(asset.yahoo_ticker, start=start.strftime("%Y-%m-%d"), end=end,
                        interval="1d", auto_adjust=False, progress=False)
        if isinstance(y.columns, pd.MultiIndex):
            y.columns = y.columns.get_level_values(0)
        y = y.reset_index()
    y["Date"] = pd.to_datetime(y["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    y = y[(y["Date"] >= start) & ((y["Date"] <= pd.Timestamp(end)) if end else True)]
    y["Source"] = "yahoo"
    return y[COLUMNS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-csv", type=Path, required=True)
    ap.add_argument("--yahoo-dir", type=Path, default=None)
    ap.add_argument("--end", default=None, help="last date to include (YYYY-MM-DD)")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--assets", nargs="*", default=list(ASSETS))
    args = ap.parse_args(argv)

    kaggle = pd.read_csv(args.kaggle_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in args.assets:
        asset = ASSETS[key]
        k = kaggle_rows(kaggle, asset.name)
        y = yahoo_rows(asset, k["Date"].max() + pd.Timedelta(days=1), args.end, args.yahoo_dir)
        out = pd.concat([k, y]).drop_duplicates("Date", keep="first").sort_values("Date")
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        path = args.out_dir / asset.csv_name
        out.to_csv(path, index=False)
        print(f"{asset.name:<9} {len(k):>5} Kaggle rows + {len(y):>5} Yahoo rows -> {path}")


if __name__ == "__main__":
    main()
