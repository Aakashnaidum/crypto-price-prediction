"""Model loading, evaluation results and optional live market data."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache

from django.conf import settings

from cryptoforecast.artifacts import bundle_path, load_bundle
from cryptoforecast.config import ASSETS

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=None)
def get_bundle(asset_key: str) -> dict:
    path = bundle_path(asset_key, settings.CRYPTO_MODELS_DIR)
    if not path.exists():
        raise ModelUnavailable(
            f"No trained model for {ASSETS[asset_key].name}. Run: python scripts/train_models.py"
        )
    return load_bundle(path)


@lru_cache(maxsize=1)
def get_results() -> dict | None:
    try:
        return json.loads(settings.CRYPTO_RESULTS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest_completed_candle(asset_key: str) -> dict | None:
    """Most recent *completed* UTC daily candle from Yahoo Finance, or None.

    Today's candle is still forming, so it is skipped.
    """
    if not settings.CRYPTO_ENABLE_LIVE_DATA:
        return None
    try:
        import yfinance as yf

        hist = yf.Ticker(ASSETS[asset_key].yahoo_ticker).history(period="10d", interval="1d", auto_adjust=False)
        hist = hist.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        today = datetime.now(timezone.utc).date()
        hist = hist[[ts.tz_convert("UTC").date() < today for ts in hist.index]]
        if hist.empty:
            return None
        row = hist.iloc[-1]
        return {
            "date": hist.index[-1].tz_convert("UTC").date().isoformat(),
            "open_price": float(row["Open"]),
            "high_price": float(row["High"]),
            "low_price": float(row["Low"]),
            "close_price": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
    except Exception:  # network/library failures must not break the page
        logger.warning("Live candle fetch failed for %s", asset_key, exc_info=True)
        return None
