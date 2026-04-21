"""Ablation study: isolate the contribution of every design decision.

Runs RF+SMOTE as the fixed baseline and knocks out one component at a time:
    * feature group (temporal / amplitude / morph only; single-lead; extended HRV)
    * resampling (none, SMOTE, borderline)
    * HMM post-processor on/off
    * drop-Q on/off
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

from src.data_loader import (AAMI_CLASSES, FEATURE_COLS, feature_group_mask,
                              load_dataset)
from src.extended_features import build_extended_matrix
from src.train import TrainConfig, cross_validate


CSV = ROOT / "MIT-BIH Arrhythmia Database.csv"
OUT_DIR = ROOT / "results" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _subset_cols(X, groups):
    idx = feature_group_mask(groups)
    return X[:, idx], idx


def run_config(X, y, groups, tag: str, **overrides):
    # ablation uses 3-fold and 150 trees — directional comparisons.
    # overrides beat defaults if the caller passes e.g. resampling="none".
    defaults = dict(model="rf", resampling="smote", n_splits=3,
                     n_estimators=150, random_state=42)
    defaults.update(overrides)
    cfg = TrainConfig(**defaults)
    t0 = time.time()
    results, preds, _ = cross_validate(X, y, groups, cfg)
    wall = time.time() - t0

    row = {"tag": tag,
           "mean_acc": float(np.mean([r.accuracy for r in results])),
           "mean_mf1": float(np.mean([r.macro_f1 for r in results])),
           "wall_s":   round(wall, 1)}
    for c in AAMI_CLASSES:
        row[f"f1_{c}"] = float(np.mean([r.per_class_f1[c] for r in results]))
    print(f"  {tag:30s}  acc={row['mean_acc']:.4f}  mF1={row['mean_mf1']:.4f}  "
          f"V={row['f1_V']:.3f}  S={row['f1_S']:.3f}  F={row['f1_F']:.3f}  "
          f"({wall:.0f}s)")
    return row


def main():
    print("Loading...")
    X, y, groups, df = load_dataset(CSV)
    rows = []

    # 0. Full baseline (all 32 features)
    print("\n[0] Baseline (all features, SMOTE)")
    rows.append(run_config(X, y, groups, "all_features_smote"))

    # 1. Single feature family
    for grp_name in ("temporal", "amplitude", "morph"):
        print(f"\n[1] Only {grp_name} features")
        Xs, _ = _subset_cols(X, [grp_name])
        rows.append(run_config(Xs, y, groups, f"only_{grp_name}"))

    # 2. Single-lead
    for lead in ("lead0", "lead1"):
        print(f"\n[2] Only {lead}")
        Xs, _ = _subset_cols(X, [lead])
        rows.append(run_config(Xs, y, groups, f"only_{lead}"))

    # 3. Without SMOTE
    print("\n[3] No SMOTE")
    rows.append(run_config(X, y, groups, "no_smote", resampling="none"))

    # 4. Borderline-SMOTE
    print("\n[4] Borderline-SMOTE")
    rows.append(run_config(X, y, groups, "borderline", resampling="borderline"))

    # 5. With HMM
    print("\n[5] SMOTE + HMM")
    rows.append(run_config(X, y, groups, "smote_hmm", hmm_smooth=True))

    # 6. Drop Q
    print("\n[6] Drop Q class")
    mask = y != "Q"
    rows.append(run_config(X[mask], y[mask], groups[mask], "drop_Q"))

    # 7. Extended feature set (HRV + cross-lead)
    print("\n[7] Extended features (base + HRV + cross-lead)")
    X_ext, cols_ext = build_extended_matrix(df)
    print(f"    -> {X_ext.shape[1]} features total")
    rows.append(run_config(X_ext, y, groups, "extended"))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "ablation.csv", index=False)
    print(f"\nWrote {ROOT / 'results' / 'ablation.csv'}")


if __name__ == "__main__":
    main()
