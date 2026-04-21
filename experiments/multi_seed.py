"""Multi-seed variance evaluation.

Same GroupKFold configuration run with 5 different random seeds; reports
mean ± std of every metric.  Answers reviewer question "is your result
a lucky seed?"
"""
from __future__ import annotations

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
SEEDS = [7, 42, 2024]


def summarize(fold_results):
    return {
        "acc":   float(np.mean([r.accuracy for r in fold_results])),
        "mf1":   float(np.mean([r.macro_f1 for r in fold_results])),
        **{f"f1_{c}": float(np.mean([r.per_class_f1[c] for r in fold_results]))
           for c in AAMI_CLASSES},
    }


def main():
    print("Loading...")
    X, y, groups, _ = load_dataset(CSV)

    rows = []
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        cfg = TrainConfig(model="rf", resampling="smote", n_splits=5,
                           n_estimators=150, random_state=seed)
        t0 = time.time()
        results, _, _ = cross_validate(X, y, groups, cfg)
        row = summarize(results)
        row["seed"] = seed
        row["wall_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  acc={row['acc']:.4f}  mF1={row['mf1']:.4f}  "
              f"V={row['f1_V']:.3f}  S={row['f1_S']:.3f}  "
              f"({row['wall_s']}s)")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "multi_seed.csv", index=False)
    print("\nPer-seed metrics:")
    print(df.to_string(index=False))

    stats = df.drop(columns=["seed", "wall_s"]).agg(["mean", "std"]).T
    stats.columns = ["mean", "std"]
    stats["mean"] = stats["mean"].round(4)
    stats["std"] = stats["std"].round(4)
    stats.to_csv(ROOT / "results" / "multi_seed_stats.csv")
    print("\nAggregated across seeds:")
    print(stats.to_string())


if __name__ == "__main__":
    main()
