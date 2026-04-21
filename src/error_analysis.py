"""Error analysis: confusion matrix, misclassification visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import AAMI_CLASSES


def per_patient_metrics(y_true, y_pred, groups) -> pd.DataFrame:
    """One row per patient: accuracy, macro-F1, per-class recall."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    y_true = np.asarray(y_true, dtype=object).astype(str)
    y_pred = np.asarray(y_pred, dtype=object).astype(str)
    groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        row = {
            "record": int(g) if str(g).isdigit() else str(g),
            "n_beats": int(mask.sum()),
            "accuracy": round(accuracy_score(yt, yp), 4),
            "macro_f1": round(f1_score(yt, yp, labels=AAMI_CLASSES,
                                        average="macro", zero_division=0), 4),
        }
        for c in AAMI_CLASSES:
            if c in yt:
                row[f"recall_{c}"] = round(recall_score(
                    yt == c, yp == c, zero_division=0), 4)
            else:
                row[f"recall_{c}"] = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1")


def plot_misclass_breakdown(y_true, y_pred, out_path):
    """Stacked bar: for each true class, fraction predicted as each other class."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    table = (df.groupby("t")["p"].value_counts(normalize=True)
               .unstack(fill_value=0.0)
               .reindex(index=AAMI_CLASSES, columns=AAMI_CLASSES, fill_value=0.0))

    fig, ax = plt.subplots(figsize=(8, 5))
    table.plot(kind="bar", stacked=True, ax=ax,
                color=["#2a9d8f", "#e9c46a", "#e76f51", "#264653", "#8d99ae"])
    ax.set_ylabel("Fraction")
    ax.set_title("Misclassification distribution per true class")
    ax.legend(title="Predicted", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_per_patient_heatmap(per_patient_df: pd.DataFrame, out_path):
    piv = per_patient_df.set_index("record")[
        [f"recall_{c}" for c in AAMI_CLASSES] + ["accuracy"]
    ].astype(float)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.2 * len(piv))))
    sns.heatmap(piv, annot=False, cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-patient performance (hardest patients on top)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
