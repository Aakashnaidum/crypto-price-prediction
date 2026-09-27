# Data

The price files are **not committed**: the licence of the Kaggle dataset could not be confirmed, and Yahoo Finance
data is subject to Yahoo's terms. Rebuild them locally:

1. Download **"Cryptocurrency Prices Data"** by maharshipandya from Kaggle
   (https://www.kaggle.com/datasets/maharshipandya/-cryptocurrency-historical-prices-dataset). It provides daily OHLCV and market cap for 56 coins,
   2013-05-05 → 2022-10-23. Check the licence shown on that page before reusing it.
2. Build the five CSVs, extending each with Yahoo Finance daily candles (`BTC-USD`, `ETH-USD`, `LTC-USD`, `XLM-USD`,
   `XRP-USD`) from 2022-10-24:

   ```bash
   pip install yfinance
   python scripts/build_dataset.py --kaggle-csv /path/to/cryptocurrency_prices_data_kaggle.csv --end 2026-05-08
   ```

   `--end 2026-05-08` matches the data used for the committed results. Leave it out to download up to the latest
   *completed* day. A day that is still in progress must not be included.

Resulting schema: `Date, Open, High, Low, Close, Volume, Source` (one row per UTC day; `Source` = `kaggle` | `yahoo`).

Provenance check: the builder was run against the saved Kaggle file and the earlier revision's saved Yahoo exports. It reproduces
the project's CSVs with identical prices on every shared date. The only differences are one missing day in the old files (2026-05-09)
and one incomplete, still-forming candle that the old refresh script appended (2026-05-10). That candle is never used as a training target.

Tests and CI do not need these files; they use `cryptoforecast/synthetic.py`.
