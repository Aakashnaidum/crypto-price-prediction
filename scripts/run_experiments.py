"""Run the forecasting study for every asset and write results/.

Usage:
    python scripts/run_experiments.py                 # all assets in data/
    python scripts/run_experiments.py --assets bitcoin xrp
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from cryptoforecast.config import ASSETS, DATA_DIR, RESULTS_DIR  # noqa: E402
from cryptoforecast.data import csv_path_for, data_quality_report, load_history  # noqa: E402
from cryptoforecast.evaluation import (  # noqa: E402
    common_frame, evaluate_holdout, evaluate_walk_forward,
)
from cryptoforecast.features import build_dataset  # noqa: E402
from cryptoforecast.models import candidate_forecasters  # noqa: E402


def fmt(x, digits=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def markdown_report(results: dict) -> str:
    lines = [
        "# Experiment results (re-run)",
        "",
        f"Generated {results['generated_utc']} with Python {results['python']}, scikit-learn {results['sklearn']}.",
        "",
        "Task: forecast the next calendar day's close using only information available at the",
        "end of the current day. Split: chronological; last 20% of usable rows = hold-out test.",
        "`MAE/Pers.` < 1 means the model beat the persistence baseline (tomorrow = today).",
        "DM p-value: Diebold-Mariano test of squared errors vs persistence (two-sided).",
        "Directional accuracy excludes days where the model predicts no move (n/a for persistence).",
        "",
    ]
    for key, r in results["assets"].items():
        h = r["holdout"]
        lines += [
            f"## {ASSETS[key].name}",
            "",
            f"Train {h['train_start']} → {h['train_end']} ({h['n_train']} days); "
            f"test {h['test_start']} → {h['test_end']} ({h['n_test']} days).",
            "",
            "| Model | MAE | RMSE | MAPE % | MAE/Pers. | Dir. acc. | DM stat | DM p | WF MAE/Pers. (folds won) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, m in sorted(h["models"].items(), key=lambda kv: kv[1]["mae"]):
            wf = r["walk_forward"][label]
            lines.append(
                f"| {label} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['mape_pct'], 2)} | "
                f"{fmt(m['mae_vs_persistence'], 3)} | {fmt(m['directional_accuracy'], 3)} | "
                f"{fmt(m['dm_stat'], 2)} | {fmt(m['p_value'], 3)} | "
                f"{fmt(wf['mean_mae_vs_persistence'], 3)} ({wf['folds_beating_persistence']}/{wf['folds']}) |"
            )
        best = r["best_learned_holdout"]
        lines += [
            "",
            f"Best learned model on hold-out: **{best['label']}** "
            f"(MAE/Pers. {fmt(best['mae_vs_persistence'], 3)}). "
            + ("It beats persistence on hold-out MAE." if best["mae_vs_persistence"] < 1
               else "No learned model beats persistence on hold-out MAE."),
            "",
        ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="*", default=list(ASSETS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    results = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "assets": {},
    }
    for key in args.assets:
        path = csv_path_for(key, args.data_dir)
        history = load_history(path)
        frame = common_frame(build_dataset(history))
        forecasters = candidate_forecasters()
        print(f"[{key}] {len(frame)} usable rows; evaluating {len(forecasters)} forecasters", flush=True)
        holdout = evaluate_holdout(frame, forecasters)
        wf = evaluate_walk_forward(frame, forecasters)
        learned = {k: v for k, v in holdout["models"].items() if k not in ("Persistence", "Drift")}
        best = min(learned, key=lambda k: learned[k]["mae"])
        results["assets"][key] = {
            "data_quality": data_quality_report(history),
            "usable_rows": int(len(frame)),
            "holdout": holdout,
            "walk_forward": wf,
            "best_learned_holdout": {"label": best, **learned[best]},
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "RESULTS.md").write_text(markdown_report(results))
    print(f"Wrote {args.out_dir / 'metrics.json'} and RESULTS.md")


if __name__ == "__main__":
    main()
