"""Evaluation and visualization helpers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import AAMI_CLASSES, FEATURE_COLS


def aggregate_confusion(fold_predictions):
    y_true = np.concatenate([np.array(f["y_true"]) for f in fold_predictions])
    y_pred = np.concatenate([np.array(f["y_pred"]) for f in fold_predictions])
    return confusion_matrix(y_true, y_pred, labels=AAMI_CLASSES), y_true, y_pred


def plot_confusion(cm, out_path, title="Confusion Matrix (patient-independent CV)"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=AAMI_CLASSES, yticklabels=AAMI_CLASSES, ax=axes[0])
    axes[0].set_title(title + " - counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=AAMI_CLASSES, yticklabels=AAMI_CLASSES, ax=axes[1])
    axes[1].set_title(title + " - row-normalized (recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_feature_importance(importances, out_path, top_n=20):
    order = np.argsort(importances)[::-1][:top_n]
    names = [FEATURE_COLS[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), importances[order][::-1], color="#3a7ca5")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean decrease in impurity (averaged over folds)")
    ax.set_title("Random Forest feature importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_fold_metrics(results, out_path):
    folds = [r.fold for r in results]
    acc = [r.accuracy for r in results]
    macro_f1 = [r.macro_f1 for r in results]
    v_f1 = [r.per_class_f1["V"] for r in results]
    s_f1 = [r.per_class_f1["S"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.2
    x = np.arange(len(folds))
    ax.bar(x - 1.5 * width, acc, width, label="Accuracy")
    ax.bar(x - 0.5 * width, macro_f1, width, label="Macro F1")
    ax.bar(x + 0.5 * width, v_f1, width, label="V-class F1")
    ax.bar(x + 1.5 * width, s_f1, width, label="S-class F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-fold metrics (patient-independent GroupKFold)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_class_distribution(y, out_path):
    from collections import Counter
    c = Counter(y)
    counts = [c.get(k, 0) for k in AAMI_CLASSES]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(AAMI_CLASSES, counts, color=["#4c956c", "#f4a261", "#e76f51", "#264653", "#9d8189"])
    ax.set_yscale("log")
    ax.set_ylabel("Beat count (log scale)")
    ax.set_title("AAMI class distribution - MIT-BIH Arrhythmia Database")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_classification_report(y_true, y_pred, out_path):
    report = classification_report(y_true, y_pred,
                                    labels=AAMI_CLASSES,
                                    target_names=AAMI_CLASSES,
                                    digits=4, zero_division=0)
    Path(out_path).write_text(report)
    return report
