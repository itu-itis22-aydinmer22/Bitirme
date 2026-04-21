"""de Chazal-style DS1 train / DS2 test single-split benchmark.

Exactly mirrors the setup of Mondéjar-Guerra 2019 / de Chazal 2004:
    - Train on records in DS1_RECORDS, test on DS2_RECORDS.
    - Report per-class F1 + aggregated confusion for every model in MATRIX.
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
OUT_DIR = ROOT / "results" / "dechazal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


MATRIX = [
    ("lda",  "none",       None,       "lda"),
    ("lda",  "smote",      None,       "lda_smote"),
    ("rf",   "smote",      None,       "rf_smote"),
    ("rf",   "borderline", None,       "rf_borderline"),
    ("lgbm", "smote",      "clinical", "lgbm_smote_cost"),
    ("xgb",  "smote",      "clinical", "xgb_smote_cost"),
    ("rf",   "smote",      None,       "rf_smote_hmm"),   # with HMM (toggled below)
]


def run_one(X, y, groups, model, resamp, weighting, hmm=False):
    cfg = TrainConfig(
        model=model, resampling=resamp, weighting=weighting,
        fold_strategy="dechazal", hmm_smooth=hmm, random_state=42,
    )
    t0 = time.time()
    results, preds, _ = cross_validate(X, y, groups, cfg)
    wall = time.time() - t0

    # single fold (DS1 train / DS2 test)
    r = results[0]
    y_true = np.array(preds[0]["y_true"])
    y_pred = np.array(preds[0]["y_pred"])
    from sklearn.metrics import accuracy_score, f1_score
    row = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=AAMI_CLASSES,
                                    average="macro", zero_division=0)),
        "wall_s":  round(wall, 1),
    }
    for c in AAMI_CLASSES:
        row[f"f1_{c}"] = r.per_class_f1[c]
    return row, y_true, y_pred


def main():
    print("Loading...")
    X, y, groups, _ = load_dataset(CSV)

    rows = []
    for model, resamp, weighting, tag in MATRIX:
        hmm = tag.endswith("_hmm")
        print(f"\n=== {tag} (hmm={hmm}) ===")
        row, y_true, y_pred = run_one(X, y, groups, model, resamp, weighting, hmm=hmm)
        row.update({"tag": tag, "model": model, "resampling": resamp,
                    "weighting": weighting or "none", "hmm": hmm})
        rows.append(row)
        np.save(OUT_DIR / f"{tag}_y_true.npy", y_true)
        np.save(OUT_DIR / f"{tag}_y_pred.npy", y_pred)
        print(f"  acc={row['accuracy']:.4f}  mF1={row['macro_f1']:.4f}  "
              f"V={row['f1_V']:.3f}  S={row['f1_S']:.3f}  F={row['f1_F']:.3f}  "
              f"({row['wall_s']}s)")

    out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out.to_csv(ROOT / "results" / "dechazal_ds1_ds2.csv", index=False)
    print(f"\nWrote {ROOT / 'results' / 'dechazal_ds1_ds2.csv'}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
