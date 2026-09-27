"""Project-wide configuration: assets, paths and experiment constants."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# All paths can be overridden with environment variables so nothing is tied to
# one machine's directory layout.
DATA_DIR = Path(os.environ.get("CRYPTO_DATA_DIR", REPO_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("CRYPTO_MODELS_DIR", REPO_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("CRYPTO_RESULTS_DIR", REPO_ROOT / "results"))


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    symbol: str
    yahoo_ticker: str

    @property
    def csv_name(self) -> str:
        return f"{self.name}.csv"


ASSETS: dict[str, Asset] = {
    a.key: a
    for a in (
        Asset("bitcoin", "Bitcoin", "BTC", "BTC-USD"),
        Asset("ethereum", "Ethereum", "ETH", "ETH-USD"),
        Asset("litecoin", "Litecoin", "LTC", "LTC-USD"),
        Asset("stellar", "Stellar", "XLM", "XLM-USD"),
        Asset("xrp", "XRP", "XRP", "XRP-USD"),
    )
}

RANDOM_STATE = 42
TEST_FRACTION = 0.2          # last 20% of usable rows (chronological) = hold-out test
VALIDATION_FRACTION = 0.2    # last 20% of the training block = model-selection split
WALK_FORWARD_FOLDS = 5
