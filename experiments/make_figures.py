"""Build the v2 comparison figures used in the final report.

Produces:
  - results/figures/model_comparison.png   (benchmark bar chart)
  - results/figures/dechazal_vs_groupkfold.png
  - results/figures/dechazal_confusion.png (aggregated XGB confusion)
  - results/figures/cross_dataset.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import AAMI_CLASSES
from sklearn.metrics import confusion_matrix


FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def plot_benchmark_bar():
    df = pd.read_csv(ROOT / "results" / "benchmark_models.csv")
    df = df.sort_values("mean_mf1", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    y = np.arange(len(df))

    axes[0].barh(y, df["mean_mf1"], color="#3a7ca5")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["tag"])
    axes[0].set_xlabel("Mean macro-F1 (3-fold GroupKFold)")
    axes[0].set_title("Benchmark: macro-F1 across 13 configurations")
    axes[0].grid(True, axis="x", alpha=0.3)

    width = 0.2
    x = np.arange(len(df))
    axes[1].barh(y - width, df["f1_V"], width, label="V-F1",   color="#e76f51")
    axes[1].barh(y,         df["f1_S"], width, label="S-F1",   color="#f4a261")
    axes[1].barh(y + width, df["f1_F"], width, label="F-F1",   color="#2a9d8f")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df["tag"])
    axes[1].set_xlabel("Per-class F1")
    axes[1].set_title("Benchmark: clinically important minority classes")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = FIGS / "model_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def plot_dechazal_vs_groupkfold():
    ds12 = pd.read_csv(ROOT / "results" / "dechazal_ds1_ds2.csv")
    bench = pd.read_csv(ROOT / "results" / "benchmark_models.csv")

    # Pick comparable configs: RF+SMOTE and XGB+SMOTE+cost
    cmp_rows = []
    for tag in ("rf_smote", "xgb_smote_cost", "lgbm_smote_cost", "lda"):
        d = ds12[ds12["tag"] == tag]
        if not d.empty:
            cmp_rows.append(("DS1/DS2 (de Chazal)", tag, float(d["accuracy"].iloc[0]),
                             float(d["macro_f1"].iloc[0]), float(d["f1_V"].iloc[0]),
                             float(d["f1_S"].iloc[0])))
    for bench_tag, label in [("rf_smote", "rf_smote"), ("xgb_cost_smote", "xgb_smote_cost"),
                              ("lgbm_cost_smote", "lgbm_smote_cost"),
                              ("lda_baseline", "lda")]:
        b = bench[bench["tag"] == bench_tag]
        if not b.empty:
            cmp_rows.append(("GroupKFold (5/3-fold)", label,
                             float(b["mean_acc"].iloc[0]),
                             float(b["mean_mf1"].iloc[0]),
                             float(b["f1_V"].iloc[0]),
                             float(b["f1_S"].iloc[0])))
    df = pd.DataFrame(cmp_rows, columns=["split", "model", "acc", "mf1", "f1_V", "f1_S"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metric_map = {"mf1": "Macro-F1", "f1_V": "V-class F1", "f1_S": "S-class F1"}
    for ax, (metric, title) in zip(axes, metric_map.items()):
        sns.barplot(data=df, x="model", y=metric, hue="split",
                     palette=["#264653", "#e76f51"], ax=ax)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        if metric != "mf1":
            ax.get_legend().remove()
    axes[0].legend(title="", loc="upper right")

    plt.tight_layout()
    out = FIGS / "dechazal_vs_groupkfold.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def plot_dechazal_confusion():
    y_true = np.load(ROOT / "results" / "dechazal" / "xgb_smote_cost_y_true.npy", allow_pickle=True)
    y_pred = np.load(ROOT / "results" / "dechazal" / "xgb_smote_cost_y_pred.npy", allow_pickle=True)
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=AAMI_CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                 xticklabels=AAMI_CLASSES, yticklabels=AAMI_CLASSES, ax=axes[0])
    axes[0].set_title("XGBoost+SMOTE+cost on DS1/DS2 — counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                 xticklabels=AAMI_CLASSES, yticklabels=AAMI_CLASSES, ax=axes[1])
    axes[1].set_title("Row-normalised (recall)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    out = FIGS / "dechazal_confusion.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def plot_cross_dataset():
    df = pd.read_csv(ROOT / "results" / "cross_dataset.csv")
    df["label"] = df["tag"] + " -> " + df["dataset"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=df, x="label", y="accuracy", ax=axes[0],
                 palette="coolwarm", hue="label", legend=False)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Cross-dataset accuracy (train: MIT-BIH)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_xlabel("")

    sns.barplot(data=df, x="label", y="f1_V", ax=axes[1],
                 palette="coolwarm", hue="label", legend=False)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Cross-dataset V-class F1")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_xlabel("")

    plt.tight_layout()
    out = FIGS / "cross_dataset.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def plot_multi_seed():
    df = pd.read_csv(ROOT / "results" / "multi_seed.csv")
    stats = pd.read_csv(ROOT / "results" / "multi_seed_stats.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["acc", "mf1", "f1_N", "f1_V", "f1_S"]
    x = np.arange(len(metrics))
    means = [stats.loc[m, "mean"] for m in metrics]
    stds  = [stats.loc[m, "std"]  for m in metrics]
    ax.bar(x, means, yerr=stds, capsize=5, color=["#264653", "#2a9d8f",
                                                    "#8ab17d", "#e76f51",
                                                    "#f4a261"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1", "F1-N", "F1-V", "F1-S"])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Multi-seed variance (seeds={list(df['seed'])})")
    ax.grid(True, axis="y", alpha=0.3)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + 0.02, f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    out = FIGS / "multi_seed.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def plot_ablation():
    csv = ROOT / "results" / "ablation.csv"
    if not csv.exists():
        print(f"  (skipped: {csv} missing)")
        return
    df = pd.read_csv(csv).copy()
    # Order rows deliberately: baseline first, then feature families, leads,
    # resampling knockouts, HMM, drop_Q, extended.
    order = ["all_features_smote", "only_temporal", "only_amplitude",
             "only_morph", "only_lead0", "only_lead1", "no_smote",
             "borderline", "smote_hmm", "drop_Q", "extended"]
    df["tag"] = pd.Categorical(df["tag"], categories=order, ordered=True)
    df = df.sort_values("tag")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(df))

    # highlight baseline
    colors = ["#264653" if t == "all_features_smote" else "#3a7ca5"
              for t in df["tag"]]
    axes[0].barh(y, df["mean_mf1"], color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["tag"])
    axes[0].set_xlabel("Mean macro-F1 (3-fold GroupKFold)")
    axes[0].set_title("Ablation: macro-F1 per knockout")
    axes[0].axvline(df[df["tag"] == "all_features_smote"]["mean_mf1"].iloc[0],
                    ls="--", color="red", lw=0.8, label="baseline")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, axis="x", alpha=0.3)

    width = 0.25
    axes[1].barh(y - width, df["f1_V"], width, label="V-F1",
                 color="#e76f51")
    axes[1].barh(y,         df["f1_S"], width, label="S-F1",
                 color="#f4a261")
    axes[1].barh(y + width, df["f1_F"], width, label="F-F1",
                 color="#2a9d8f")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df["tag"])
    axes[1].set_xlabel("Per-class F1")
    axes[1].set_title("Ablation: minority-class F1 per knockout")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = FIGS / "ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("Generating figures...")
    plot_benchmark_bar()
    plot_dechazal_vs_groupkfold()
    plot_dechazal_confusion()
    plot_cross_dataset()
    plot_multi_seed()
    plot_ablation()


if __name__ == "__main__":
    main()
