"""WP3 - Random Forest training with SMOTE + GroupKFold (patient-independent CV)."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .data_loader import AAMI_CLASSES


@dataclass
class TrainConfig:
    n_estimators: int = 200
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    n_jobs: int = -1
    random_state: int = 42
    class_weight: str = "balanced_subsample"
    use_smote: bool = True
    smote_k_neighbors: int = 5
    n_splits: int = 5


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict
    confusion: list
    train_seconds: float
    inference_ms_per_beat: float


def build_classifier(cfg: TrainConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        class_weight=cfg.class_weight,
    )


def _smote_safe(X, y, cfg: TrainConfig):
    """SMOTE that adapts k_neighbors when minority classes are tiny.
    Drops classes with fewer than 2 samples (cannot synthesize)."""
    counts = {c: int(np.sum(y == c)) for c in np.unique(y)}
    min_count = min(counts.values())
    if min_count < 2:
        keep = [c for c, n in counts.items() if n >= 2]
        mask = np.isin(y, keep)
        X, y = X[mask], y[mask]
        counts = {c: int(np.sum(y == c)) for c in np.unique(y)}
        min_count = min(counts.values())

    k = max(1, min(cfg.smote_k_neighbors, min_count - 1))
    smote = SMOTE(random_state=cfg.random_state, k_neighbors=k)
    return smote.fit_resample(X, y)


def cross_validate(X, y, groups, cfg: TrainConfig | None = None):
    """Patient-independent GroupKFold with optional SMOTE on the training fold only."""
    cfg = cfg or TrainConfig()
    gkf = GroupKFold(n_splits=cfg.n_splits)
    results: list[FoldResult] = []
    fold_predictions = []
    feature_importances = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if cfg.use_smote:
            X_tr_s, y_tr = _smote_safe(X_tr_s, y_tr, cfg)

        clf = build_classifier(cfg)
        t0 = time.time()
        clf.fit(X_tr_s, y_tr)
        train_seconds = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_te_s)
        inference_ms = (time.time() - t0) / max(1, len(X_te_s)) * 1000.0

        labels_present = sorted(set(y_te) | set(y_pred))
        per_class = f1_score(y_te, y_pred, labels=AAMI_CLASSES,
                             average=None, zero_division=0)
        per_class_f1 = {c: float(v) for c, v in zip(AAMI_CLASSES, per_class)}
        cm = confusion_matrix(y_te, y_pred, labels=AAMI_CLASSES).tolist()

        results.append(FoldResult(
            fold=fold,
            accuracy=float(accuracy_score(y_te, y_pred)),
            macro_f1=float(f1_score(y_te, y_pred, labels=AAMI_CLASSES,
                                     average="macro", zero_division=0)),
            per_class_f1=per_class_f1,
            confusion=cm,
            train_seconds=train_seconds,
            inference_ms_per_beat=inference_ms,
        ))
        fold_predictions.append({
            "fold": fold,
            "test_groups": sorted(set(groups[test_idx].tolist())),
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
        })
        feature_importances.append(clf.feature_importances_)

    return results, fold_predictions, np.mean(feature_importances, axis=0)


def fit_full_model(X, y, cfg: TrainConfig | None = None):
    """Final model trained on the full dataset (with SMOTE) for deployment."""
    cfg = cfg or TrainConfig()
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    if cfg.use_smote:
        X_s, y = _smote_safe(X_s, y, cfg)
    clf = build_classifier(cfg)
    clf.fit(X_s, y)
    return clf, scaler


def save_model(clf, scaler, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "rf_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")


def serialize_results(results, out_path):
    payload = [r.__dict__ for r in results]
    Path(out_path).write_text(json.dumps(payload, indent=2))
