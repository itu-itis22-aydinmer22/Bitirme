"""Benchmark: every (model × resampling × weighting) combination on MIT-BIH GroupKFold.

Writes ``results/benchmark_models.csv`` + per-config JSON under ``results/benchmark/``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import AAMI_CLASSES, load_dataset
from src.train import TrainConfig, cross_validate


CSV = ROOT / "MIT-BIH Arrhythmia Database.csv"
OUT_DIR = ROOT / "results" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# The grid is deliberately tight for runtime budget (each fit is ~1-2 min).
MATRIX = [
    # model,  resampling,   weighting,  drop_q, hmm,   tag
    ("rf",    "none",       "balanced", False,  False, "rf_cw"),
    ("rf",    "smote",      None,       False,  False, "rf_smote"),
    ("rf",    "borderline", None,       False,  False, "rf_borderline"),
    ("rf",    "adasyn",     None,       False,  False, "rf_adasyn"),
    ("rf",    "smote",      None,       True,   False, "rf_smote_dropQ"),
    ("rf",    "smote",      None,       False,  True,  "rf_smote_hmm"),
    ("lgbm",  "none",       "clinical", False,  False, "lgbm_cost"),
    ("lgbm",  "smote",      "clinical", False,  False, "lgbm_cost_smote"),
    ("lgbm",  "borderline", "clinical", False,  False, "lgbm_cost_borderline"),
    ("xgb",   "none",       "clinical", False,  False, "xgb_cost"),
    ("xgb",   "smote",      "clinical", False,  False, "xgb_cost_smote"),
    ("lda",   "none",       None,       False,  False, "lda_baseline"),
    ("lda",   "smote",      None,       False,  False, "lda_smote"),
]


def summarize(results, preds):
    y_true = np.concatenate([np.array(f["y_true"]) for f in preds])
    y_pred = np.concatenate([np.array(f["y_pred"]) for f in preds])

    from sklearn.metrics import accuracy_score, f1_score
    row = {
        "mean_acc":  float(np.mean([r.accuracy  for r in results])),
        "mean_mf1":  float(np.mean([r.macro_f1  for r in results])),
        "acc_std":   float(np.std([r.accuracy  for r in results])),
        "mf1_std":   float(np.std([r.macro_f1  for r in results])),
        "agg_acc":   float(accuracy_score(y_true, y_pred)),
        "agg_mf1":   float(f1_score(y_true, y_pred, labels=AAMI_CLASSES,
                                      average="macro", zero_division=0)),
        "train_s":   float(np.mean([r.train_seconds for r in results])),
        "ms_beat":   float(np.mean([r.inference_ms_per_beat for r in results])),
    }
    for c in AAMI_CLASSES:
        row[f"f1_{c}"] = float(np.mean([r.per_class_f1[c] for r in results]))
    return row, y_true, y_pred


def main():
    print("Loading data...")
    X, y, groups, df = load_dataset(CSV)
    print(f"  -> {X.shape[0]:,} beats, {X.shape[1]} features")

    rows = []
    for model, resamp, weighting, drop_q, hmm, tag in MATRIX:
        print(f"\n=== {tag}: model={model} resamp={resamp} w={weighting} "
              f"dropQ={drop_q} hmm={hmm} ===")
        X_run, y_run, g_run = X, y, groups
        if drop_q:
            mask = y != "Q"
            X_run, y_run, g_run = X[mask], y[mask], groups[mask]

        cfg = TrainConfig(
            model=model, resampling=resamp, weighting=weighting,
            hmm_smooth=hmm, drop_q=drop_q, n_splits=3, n_estimators=150,
            random_state=42,
        )
        t0 = time.time()
        results, preds, imp = cross_validate(X_run, y_run, g_run, cfg)
        wall = time.time() - t0

        row, y_true, y_pred = summarize(results, preds)
        row.update({"tag": tag, "model": model, "resampling": resamp,
                    "weighting": weighting or "none",
                    "drop_q": drop_q, "hmm": hmm, "wall_s": round(wall, 1)})
        rows.append(row)
        print(f"  -> acc={row['mean_acc']:.4f}  mF1={row['mean_mf1']:.4f}  "
              f"F1-V={row['f1_V']:.3f}  F1-S={row['f1_S']:.3f}  "
              f"F1-F={row['f1_F']:.3f}  ({wall:.0f}s)")

        with open(OUT_DIR / f"{tag}.json", "w") as fh:
            json.dump({"summary": row, "folds": [r.__dict__ for r in results]},
                      fh, indent=2)

    out = pd.DataFrame(rows)
    out = out.sort_values("mean_mf1", ascending=False)
    col_order = (["tag", "model", "resampling", "weighting", "drop_q", "hmm",
                  "mean_acc", "mean_mf1", "agg_mf1"]
                 + [f"f1_{c}" for c in AAMI_CLASSES]
                 + ["acc_std", "mf1_std", "train_s", "ms_beat", "wall_s"])
    out[col_order].to_csv(ROOT / "results" / "benchmark_models.csv", index=False)
    print(f"\nWrote {ROOT / 'results' / 'benchmark_models.csv'}")
    print(out[col_order[:9]].to_string(index=False))


if __name__ == "__main__":
    main()
