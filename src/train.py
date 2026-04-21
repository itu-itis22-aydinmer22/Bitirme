"""Multi-model, multi-resampler patient-independent CV driver.

Supports:
    - any of {rf, lgbm, xgb, lda, logreg, stack}
    - any of {none, smote, borderline, adasyn, smotetomek, smoteenn, undersample}
    - optional HMM post-processing on the predicted probabilities
    - optional drop-Q mode (Q has only 15 samples — often not learnable)
    - optional cost-sensitive sample_weight (``weighting`` ∈ {None, 'balanced', 'clinical'})
    - fold strategies: ``groupkfold`` (stratified over patients) or
      ``dechazal`` (single deterministic DS1 train / DS2 test split).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import json
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .data_loader import AAMI_CLASSES, DS1_RECORDS, DS2_RECORDS
from .models import build_model, fit_with_weights, LabelSafeXGB
from .postprocessing import estimate_transition_matrix, viterbi_smooth
from .resampling import resample


# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model: str = "rf"
    resampling: str = "smote"
    weighting: str | None = "clinical"   # None | 'balanced' | 'clinical'
    n_estimators: int = 200
    n_splits: int = 5
    fold_strategy: str = "groupkfold"    # or 'dechazal'
    random_state: int = 42
    drop_q: bool = False
    hmm_smooth: bool = False
    hmm_self_prior: float = 0.85
    # passthrough for RF backward compatibility
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    n_jobs: int = -1
    class_weight: str = "balanced_subsample"
    use_smote: bool = True               # kept for compat; drives resampling when True
    smote_k_neighbors: int = 5


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict
    confusion: list
    train_seconds: float
    inference_ms_per_beat: float


# ---------------------------------------------------------------------------

def _resolve_resampling(cfg: TrainConfig) -> str:
    # Back-compat: old TrainConfig used `use_smote: bool`.
    # If `resampling` is explicitly set (!= "smote") respect it;
    # otherwise let `use_smote` toggle between "smote" and "none".
    if cfg.resampling != "smote":
        return cfg.resampling
    return "smote" if cfg.use_smote else "none"


def _build_folds(X, y, groups, cfg: TrainConfig):
    if cfg.fold_strategy == "dechazal":
        tr = np.isin(groups, DS1_RECORDS)
        te = np.isin(groups, DS2_RECORDS)
        return [(np.where(tr)[0], np.where(te)[0])]
    gkf = GroupKFold(n_splits=cfg.n_splits)
    return list(gkf.split(X, y, groups))


def _proba_with_class_order(model, X, class_order=AAMI_CLASSES):
    """Return (N, |class_order|) — filling missing classes with zeros."""
    proba = model.predict_proba(X)
    classes = list(map(str, model.classes_))
    out = np.zeros((len(X), len(class_order)), dtype=np.float32)
    for i, c in enumerate(class_order):
        if c in classes:
            out[:, i] = proba[:, classes.index(c)]
    # Normalize rows so they sum to 1 (numerical safety).
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    out = out / row_sum
    return out


def build_classifier(cfg: TrainConfig):
    """Legacy helper kept for backwards compatibility with old main.py."""
    if cfg.model == "rf":
        return build_model(
            "rf",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            class_weight=cfg.class_weight,
        )
    return build_model(cfg.model, n_estimators=cfg.n_estimators,
                       random_state=cfg.random_state)


# ---------------------------------------------------------------------------

def cross_validate(X, y, groups, cfg: TrainConfig | None = None):
    """Run the configured fold strategy and return (results, preds, mean_imp)."""
    cfg = cfg or TrainConfig()
    resampling_strategy = _resolve_resampling(cfg)
    folds = _build_folds(X, y, groups, cfg)

    # Optional: learn HMM transitions from the training folds' labels
    transitions = None
    if cfg.hmm_smooth:
        all_train_y = np.concatenate([y[tr] for tr, _ in folds])
        all_train_g = np.concatenate([groups[tr] for tr, _ in folds])
        transitions = estimate_transition_matrix(
            all_train_y, all_train_g, self_prior=cfg.hmm_self_prior,
        )

    results: list[FoldResult] = []
    fold_predictions = []
    feature_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if resampling_strategy not in ("none", None):
            X_tr_s, y_tr = resample(X_tr_s, y_tr, resampling_strategy,
                                     random_state=cfg.random_state,
                                     default_k=cfg.smote_k_neighbors)

        clf = build_classifier(cfg) if cfg.model == "rf" else build_model(
            cfg.model, random_state=cfg.random_state,
        )

        t0 = time.time()
        clf = fit_with_weights(clf, X_tr_s, y_tr, weighting=cfg.weighting)
        train_seconds = time.time() - t0

        t0 = time.time()
        if cfg.hmm_smooth:
            proba = _proba_with_class_order(clf, X_te_s)
            y_pred = viterbi_smooth(proba, groups[test_idx], transitions)
        else:
            y_pred = clf.predict(X_te_s)
        inference_ms = (time.time() - t0) / max(1, len(X_te_s)) * 1000.0
        y_pred = np.asarray(y_pred, dtype=object).astype(str)

        per_class = f1_score(y_te, y_pred, labels=AAMI_CLASSES,
                             average=None, zero_division=0)
        per_class_f1 = {c: float(v) for c, v in zip(AAMI_CLASSES, per_class)}
        cm = confusion_matrix(y_te, y_pred, labels=AAMI_CLASSES).tolist()

        results.append(FoldResult(
            fold=fold_idx,
            accuracy=float(accuracy_score(y_te, y_pred)),
            macro_f1=float(f1_score(y_te, y_pred, labels=AAMI_CLASSES,
                                     average="macro", zero_division=0)),
            per_class_f1=per_class_f1,
            confusion=cm,
            train_seconds=train_seconds,
            inference_ms_per_beat=inference_ms,
        ))
        fold_predictions.append({
            "fold": fold_idx,
            "test_groups": sorted(set(groups[test_idx].tolist())),
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
        })
        if hasattr(clf, "feature_importances_"):
            fi = np.asarray(clf.feature_importances_, dtype=np.float32)
            if fi.shape[0] == X.shape[1]:
                feature_importances.append(fi)

    mean_imp = (np.mean(feature_importances, axis=0)
                if feature_importances else np.zeros(X.shape[1], dtype=np.float32))
    return results, fold_predictions, mean_imp


def fit_full_model(X, y, cfg: TrainConfig | None = None):
    """Train a deployment model on all data."""
    cfg = cfg or TrainConfig()
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    strategy = _resolve_resampling(cfg)
    if strategy not in ("none", None):
        X_s, y = resample(X_s, y, strategy,
                           random_state=cfg.random_state,
                           default_k=cfg.smote_k_neighbors)
    clf = build_classifier(cfg) if cfg.model == "rf" else build_model(
        cfg.model, random_state=cfg.random_state,
    )
    clf = fit_with_weights(clf, X_s, y, weighting=cfg.weighting)
    return clf, scaler


# ---------------------------------------------------------------------------
# Compatibility helpers retained for the original main.py entry point.

def _smote_safe(X, y, cfg: TrainConfig):
    # Back-compat shim: old main.py imported this symbol.
    return resample(X, y, "smote",
                     random_state=cfg.random_state,
                     default_k=cfg.smote_k_neighbors)


def save_model(clf, scaler, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name by model type if possible; fall back to 'model'
    name = type(clf).__name__.lower()
    joblib.dump(clf, out_dir / f"{name}_model.joblib")
    joblib.dump(clf, out_dir / "rf_model.joblib")  # compat name
    joblib.dump(scaler, out_dir / "scaler.joblib")


def serialize_results(results, out_path):
    payload = [r.__dict__ for r in results]
    Path(out_path).write_text(json.dumps(payload, indent=2))
