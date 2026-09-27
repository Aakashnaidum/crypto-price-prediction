"""Train the model bundles served by the web app.

For each asset:
  1. choose among single-candle models using ONLY the training block
     (inner chronological validation split),
  2. score the chosen model and persistence on the untouched hold-out test,
  3. refit the chosen model on all rows and save models/<asset>.joblib with
     that evidence embedded, so the app can show it next to every forecast.

Usage: python scripts/train_models.py [--assets bitcoin ...] [--data-dir DIR] [--models-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptoforecast.artifacts import bundle_path, save_bundle  # noqa: E402
from cryptoforecast.config import ASSETS, DATA_DIR, MODELS_DIR  # noqa: E402
from cryptoforecast.data import csv_path_for, load_history  # noqa: E402
from cryptoforecast.evaluation import (  # noqa: E402
    chronological_split, common_frame, evaluate_holdout, select_on_validation,
)
from cryptoforecast.features import build_dataset  # noqa: E402
from cryptoforecast.models import Persistence, deployable_forecasters  # noqa: E402


def train_asset(key: str, data_dir: Path, models_dir: Path) -> dict:
    frame = common_frame(build_dataset(load_history(csv_path_for(key, data_dir))))
    candidates = deployable_forecasters()
    train, _ = chronological_split(frame)
    chosen_label, val_scores = select_on_validation(train, candidates)
    chosen = next(m for m in candidates if m.label == chosen_label)

    holdout = evaluate_holdout(frame, [Persistence(), chosen])
    model_m = holdout["models"][chosen_label]
    pers_m = holdout["models"]["Persistence"]
    evidence = {
        "selection": "lowest validation MAE inside the training block",
        "validation_mae": val_scores,
        "test_period": f"{holdout['test_start']} to {holdout['test_end']}",
        "n_test": holdout["n_test"],
        "model_mae": model_m["mae"],
        "persistence_mae": pers_m["mae"],
        "mae_vs_persistence": model_m["mae_vs_persistence"],
        "directional_accuracy": model_m["directional_accuracy"],
        "dm_p_value": model_m["p_value"],
        "beats_persistence": model_m["mae_vs_persistence"] < 1,
    }
    final = deepcopy(chosen).fit(frame)
    trained_through = str(frame["next_date"].iloc[-1].date())
    save_bundle(bundle_path(key, models_dir), asset_key=key, model=final,
                trained_through=trained_through, evidence=evidence)
    return {"model": chosen_label, "trained_through": trained_through, **evidence}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", nargs="*", default=list(ASSETS))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)
    summary = {k: train_asset(k, args.data_dir, args.models_dir) for k in args.assets}
    for k, s in summary.items():
        print(f"{k:<9} {s['model']:<28} MAE/persistence={s['mae_vs_persistence']:.3f} "
              f"beats_persistence={s['beats_persistence']}")
    (args.models_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
