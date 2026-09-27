"""Re-derive the saved historical summary (results/legacy/training_summary.json).

The saved file was produced by an earlier version of this project whose protocol
differs from the main study:
  * target = Close of the *next row* in the CSV (not necessarily the next calendar day)
  * features = same-day High, Low, Open, Volume; target in price levels
  * chronological 80/20 split over all rows

This script repeats that protocol on the current CSVs and prints the saved vs
re-computed hold-out RMSE for the learned models and the persistence baseline.

Usage: python scripts/verify_saved_summary.py [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from cryptoforecast.config import DATA_DIR, REPO_ROOT  # noqa: E402
from cryptoforecast.data import csv_path_for, load_history  # noqa: E402

LEGACY_MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForestRegressor": lambda: RandomForestRegressor(
        random_state=42, n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1),
    "DecisionTreeRegressor": lambda: DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_leaf=5),
}
FEATURES = ["High", "Low", "Open", "Volume"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def recompute(history):
    d = history.copy()
    d["NextClose"] = d["Close"].shift(-1)
    d = d.dropna(subset=["NextClose"]).reset_index(drop=True)
    cut = int(len(d) * 0.8)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    out = {"PersistenceBaseline": rmse(te["Close"], te["NextClose"])}
    for name, make in LEGACY_MODELS.items():
        model = make().fit(tr[FEATURES], tr["NextClose"])
        out[name] = rmse(model.predict(te[FEATURES]), te["NextClose"])
    return out, len(d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--saved", type=Path, default=REPO_ROOT / "results" / "legacy" / "training_summary.json")
    args = ap.parse_args(argv)
    saved = json.loads(args.saved.read_text())

    report = {}
    print(f"{'asset':<9} {'model':<22} {'saved RMSE':>14} {'recomputed':>14} {'rel.diff':>9}  match(<0.5%)")
    for asset, s in saved.items():
        rec, rows = recompute(load_history(csv_path_for(asset, args.data_dir)))
        report[asset] = {"rows_saved": s["rows"], "rows_recomputed": rows, "models": {}}
        for model, m in s["candidate_results"].items():
            rel = abs(rec[model] - m["rmse"]) / m["rmse"]
            ok = rel < 5e-3  # RF/threads/library versions cause tiny drift
            report[asset]["models"][model] = {"saved": m["rmse"], "recomputed": rec[model], "relative_diff": rel, "match": ok}
            print(f"{asset:<9} {model:<22} {m['rmse']:>14.6g} {rec[model]:>14.6g} {rel:>9.1e}  {'yes' if ok else 'NO'}")
        best_learned = min((k for k in rec if k != "PersistenceBaseline"), key=rec.get)
        report[asset]["persistence_beats_best_learned"] = rec["PersistenceBaseline"] < rec[best_learned]
    print(json.dumps({a: r["persistence_beats_best_learned"] for a, r in report.items()}, indent=2))
    out = REPO_ROOT / "results" / "legacy" / "verification.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return report


if __name__ == "__main__":
    main()
