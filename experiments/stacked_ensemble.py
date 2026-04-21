"""Stacked ensemble CV evaluation (RF + LGBM + XGB -> logistic meta).

Uses GroupKFold so the meta-learner's training data is patient-disjoint
from the test fold (honest stacking).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import AAMI_CLASSES, load_dataset
from src.models import build_model, fit_with_weights
from src.resampling import resample


CSV = ROOT / "MIT-BIH Arrhythmia Database.csv"


def _proba_with_order(model, X, class_order=AAMI_CLASSES):
    p = model.predict_proba(X)
    classes = list(map(str, model.classes_))
    out = np.zeros((X.shape[0], len(class_order)), dtype=np.float32)
    for i, c in enumerate(class_order):
        if c in classes:
            out[:, i] = p[:, classes.index(c)]
    return out


def main():
    print("Loading...")
    X, y, groups, _ = load_dataset(CSV)

    bases = ("rf", "lgbm", "xgb")
    gkf = GroupKFold(n_splits=5)

    fold_rows = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        print(f"\n=== Fold {fold} ===")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr = groups[train_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Inner CV to build honest meta features
        inner = GroupKFold(n_splits=3)
        oof = {name: np.zeros((len(y_tr), len(AAMI_CLASSES)), dtype=np.float32)
               for name in bases}

        t0 = time.time()
        for inner_fold, (i_tr, i_te) in enumerate(inner.split(X_tr_s, y_tr, g_tr)):
            X_i_tr, y_i_tr = X_tr_s[i_tr], y_tr[i_tr]
            X_i_tr, y_i_tr = resample(X_i_tr, y_i_tr, "smote", random_state=42)
            for name in bases:
                m = build_model(name)
                m = fit_with_weights(m, X_i_tr, y_i_tr, weighting="clinical")
                oof[name][i_te] = _proba_with_order(m, X_tr_s[i_te])
        print(f"  inner OOF done in {time.time()-t0:.0f}s")

        # Fit bases on full outer train, predict on test
        X_full, y_full = resample(X_tr_s, y_tr, "smote", random_state=42)
        test_probs = []
        for name in bases:
            m = build_model(name)
            m = fit_with_weights(m, X_full, y_full, weighting="clinical")
            test_probs.append(_proba_with_order(m, X_te_s))

        Z_train = np.hstack([oof[n] for n in bases])
        Z_test  = np.hstack(test_probs)

        from sklearn.linear_model import LogisticRegression
        meta = LogisticRegression(C=1.0, max_iter=3000, multi_class="multinomial",
                                    class_weight="balanced", n_jobs=-1,
                                    random_state=42)
        meta.fit(Z_train, y_tr)
        y_pred = meta.predict(Z_test)

        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, labels=AAMI_CLASSES,
                        average="macro", zero_division=0)
        per_class = f1_score(y_te, y_pred, labels=AAMI_CLASSES,
                              average=None, zero_division=0)
        row = {"fold": fold, "acc": acc, "mf1": mf1}
        for c, v in zip(AAMI_CLASSES, per_class):
            row[f"f1_{c}"] = v
        fold_rows.append(row)
        print(f"  stacked: acc={acc:.4f} mF1={mf1:.4f}")
        all_true.append(y_te)
        all_pred.append(y_pred)

    df = pd.DataFrame(fold_rows)
    df.to_csv(ROOT / "results" / "stacked_folds.csv", index=False)

    y_t = np.concatenate(all_true)
    y_p = np.concatenate(all_pred)
    cm = confusion_matrix(y_t, y_p, labels=AAMI_CLASSES)
    np.savetxt(ROOT / "results" / "stacked_confusion.csv",
                cm, fmt="%d", delimiter=",",
                header=",".join(AAMI_CLASSES), comments="")

    summary = {
        "mean_acc": float(df["acc"].mean()),
        "mean_mf1": float(df["mf1"].mean()),
        **{f"f1_{c}": float(df[f"f1_{c}"].mean()) for c in AAMI_CLASSES},
    }
    import json
    (ROOT / "results" / "stacked_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nStacked ensemble summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
