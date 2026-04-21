"""Train on MIT-BIH, test on other PhysioNet databases (INCART, SVDB).

Goal: quantify how well the classifier generalises across acquisition setups.

Skipping SCDH because its CSV contains malformed rows and the label
vocabulary differs; including it would contaminate the AAMI mapping.
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

from src.data_loader import AAMI_CLASSES, FEATURE_COLS, load_dataset
from src.train import TrainConfig, fit_full_model
from sklearn.metrics import (accuracy_score, classification_report,
                               confusion_matrix, f1_score)


TRAIN_CSV = ROOT / "MIT-BIH Arrhythmia Database.csv"
TEST_CSVS = {
    "SVDB":   ROOT / "MIT-BIH Supraventricular Arrhythmia Database.csv",
    "INCART": ROOT / "INCART 2-lead Arrhythmia Database.csv",
}


def evaluate(model, scaler, X, y):
    X_s = scaler.transform(X)
    y_pred = model.predict(X_s)
    row = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=AAMI_CLASSES,
                                    average="macro", zero_division=0)),
    }
    for c in AAMI_CLASSES:
        # Only include F1 for classes that actually exist in y
        if c in set(y):
            mask = np.isin(y, [c]) | np.isin(y_pred, [c])
            row[f"f1_{c}"] = float(f1_score(y == c, y_pred == c,
                                              zero_division=0))
        else:
            row[f"f1_{c}"] = None
    row["confusion"] = confusion_matrix(y, y_pred, labels=AAMI_CLASSES).tolist()
    row["n"] = int(len(y))
    return row, y_pred


def main():
    print("Loading MIT-BIH (train source)...")
    X_tr, y_tr, g_tr, _ = load_dataset(TRAIN_CSV)

    print("Fitting deployment models (RF+SMOTE, LGBM+cost)...")
    rows = []
    for tag, cfg in [
        ("rf_smote",   TrainConfig(model="rf",   resampling="smote",    weighting=None)),
        ("lgbm_cost",  TrainConfig(model="lgbm", resampling="smote",    weighting="clinical")),
    ]:
        t0 = time.time()
        model, scaler = fit_full_model(X_tr, y_tr, cfg)
        print(f"  {tag}: fitted in {time.time()-t0:.1f}s")

        for db, csv in TEST_CSVS.items():
            if not csv.exists():
                print(f"  {db}: missing CSV, skipped")
                continue
            try:
                X_te, y_te, _, _ = load_dataset(csv)
            except Exception as e:
                print(f"  {db}: load failed ({e})")
                continue
            if X_te.shape[1] != X_tr.shape[1]:
                print(f"  {db}: feature dim {X_te.shape[1]} != train {X_tr.shape[1]}")
                continue

            row, y_pred = evaluate(model, scaler, X_te, y_te)
            row.update({"tag": tag, "dataset": db})
            print(f"    {tag} -> {db}: acc={row['accuracy']:.4f}  "
                  f"mF1={row['macro_f1']:.4f}  n={row['n']:,}")
            rows.append(row)

    # Drop confusion matrix for CSV, keep JSON
    (ROOT / "results" / "cross_dataset.json").write_text(json.dumps(rows, indent=2))
    csv_rows = []
    for r in rows:
        cr = {k: v for k, v in r.items() if k not in ("confusion",)}
        csv_rows.append(cr)
    pd.DataFrame(csv_rows).to_csv(ROOT / "results" / "cross_dataset.csv", index=False)
    print(f"\nWrote {ROOT / 'results' / 'cross_dataset.csv'}")


if __name__ == "__main__":
    main()
