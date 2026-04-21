"""Stacked ensemble: RF + LGBM + XGB base learners, LogReg meta-learner.

Each base model sees the same feature vector; their *out-of-fold* predicted
probabilities are concatenated and fed into a logistic-regression meta-learner.
Meta-learner is trained on the held-out predictions so the stacking is honest
about generalisation (no leakage from base -> meta on the same patients).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .models import LabelSafeXGB, build_model, fit_with_weights


def oof_probabilities(base_name, X, y, folds, weighting="clinical"):
    """Produce (N, K) out-of-fold probability matrix for one base learner."""
    probs = None
    classes = None
    for train_idx, test_idx in folds:
        model = build_model(base_name)
        model = fit_with_weights(model, X[train_idx], y[train_idx],
                                  weighting=weighting)
        p = model.predict_proba(X[test_idx])
        if classes is None:
            classes = list(map(str, model.classes_))
            probs = np.zeros((len(y), p.shape[1]), dtype=np.float32)
        probs[test_idx] = p
    return probs, classes


def fit_stacked(X, y, folds, bases=("rf", "lgbm", "xgb"),
                 weighting="clinical", random_state=42):
    meta_cols = []
    meta_class_layouts = []
    for name in bases:
        p, classes = oof_probabilities(name, X, y, folds, weighting=weighting)
        meta_cols.append(p)
        meta_class_layouts.append(classes)

    Z = np.hstack(meta_cols)
    meta = LogisticRegression(C=1.0, max_iter=3000, multi_class="multinomial",
                               class_weight="balanced",
                               random_state=random_state, n_jobs=-1)
    meta.fit(Z, y)
    return meta, meta_class_layouts


def predict_stacked(X, fitted_bases, meta, meta_class_layouts):
    """Utility: given an already-fitted list of base models and meta learner."""
    cols = [m.predict_proba(X) for m in fitted_bases]
    Z = np.hstack(cols)
    return meta.predict(Z)
