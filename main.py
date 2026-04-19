"""End-to-end runner for the ITU graduation project pipeline.

Runs:
  1. Data loading + AAMI mapping + class distribution
  2. 5-fold patient-independent cross-validation (RF + SMOTE)
  3. Aggregated confusion matrix + per-class metrics + plots
  4. Final model trained on full data, persisted to results/models/
  5. Robustness experiment under additive noise (per Sec. 6.2 of the report)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    AAMI_CLASSES, FEATURE_COLS, class_distribution, load_dataset,
)
from src.train import (
    TrainConfig, cross_validate, fit_full_model, save_model, serialize_results,
)
from src.evaluate import (
    aggregate_confusion, plot_class_distribution, plot_confusion,
    plot_feature_importance, plot_fold_metrics, write_classification_report,
)
from src.robustness import evaluate_under_noise


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = RESULTS / "models"


def main(csv_path: Path, n_estimators: int, n_splits: int, no_smote: bool):
    print(f"[1/5] Loading {csv_path.name}")
    X, y, groups, df = load_dataset(csv_path)
    dist = class_distribution(y)
    print("Class distribution (AAMI 5-class):")
    print(dist.to_string())
    dist.to_csv(RESULTS / "class_distribution.csv")
    plot_class_distribution(y, FIGS / "class_distribution.png")

    print(f"\n[2/5] Patient-independent {n_splits}-fold CV with Random Forest (n_estimators={n_estimators}, SMOTE={not no_smote})")
    cfg = TrainConfig(n_estimators=n_estimators, n_splits=n_splits, use_smote=not no_smote)
    fold_results, fold_predictions, mean_importance = cross_validate(X, y, groups, cfg)

    summary = pd.DataFrame([{
        "fold": r.fold,
        "accuracy": round(r.accuracy, 4),
        "macro_f1": round(r.macro_f1, 4),
        **{f"f1_{c}": round(r.per_class_f1[c], 4) for c in AAMI_CLASSES},
        "train_seconds": round(r.train_seconds, 2),
        "ms_per_beat": round(r.inference_ms_per_beat, 4),
    } for r in fold_results])
    print("\nPer-fold metrics:")
    print(summary.to_string(index=False))

    mean_row = {
        "fold": "mean",
        "accuracy": round(summary["accuracy"].mean(), 4),
        "macro_f1": round(summary["macro_f1"].mean(), 4),
        **{f"f1_{c}": round(summary[f"f1_{c}"].mean(), 4) for c in AAMI_CLASSES},
        "train_seconds": round(summary["train_seconds"].mean(), 2),
        "ms_per_beat": round(summary["ms_per_beat"].mean(), 4),
    }
    summary_with_mean = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    summary_with_mean.to_csv(RESULTS / "fold_metrics.csv", index=False)
    serialize_results(fold_results, RESULTS / "fold_metrics_full.json")
    print("\nMean across folds:", mean_row)

    print("\n[3/5] Aggregated evaluation + plots")
    cm, y_true, y_pred = aggregate_confusion(fold_predictions)
    report = write_classification_report(y_true, y_pred, RESULTS / "classification_report.txt")
    print(report)
    plot_confusion(cm, FIGS / "confusion_matrix.png")
    plot_feature_importance(mean_importance, FIGS / "feature_importance.png")
    plot_fold_metrics(fold_results, FIGS / "fold_metrics.png")
    np.savetxt(RESULTS / "feature_importance.csv",
               np.column_stack([np.array(FEATURE_COLS), mean_importance]),
               fmt="%s", delimiter=",", header="feature,importance", comments="")

    print("\n[4/5] Training final model on full dataset")
    final_clf, final_scaler = fit_full_model(X, y, cfg)
    save_model(final_clf, final_scaler, MODELS)
    print(f"Saved model to {MODELS}/")

    print("\n[5/5] Robustness experiment (AWGN at multiple SNR levels)")
    rob = evaluate_under_noise(final_clf, final_scaler, X, y,
                                snr_levels=(20, 15, 10, 5, 0))
    print(json.dumps(rob, indent=2))
    (RESULTS / "robustness.json").write_text(json.dumps(rob, indent=2))

    print("\nDone. Outputs under results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "MIT-BIH Arrhythmia Database.csv")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    main(args.csv, args.n_estimators, args.n_splits, args.no_smote)
