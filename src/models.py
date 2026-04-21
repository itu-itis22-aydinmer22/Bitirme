"""Unified model factory for the arrhythmia pipeline.

Wraps scikit-learn, LightGBM, XGBoost and LDA behind a common
``build_model(name, y_train, **kwargs)`` interface that returns a
fitted-ready estimator with AAMI-aware cost-sensitive defaults.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .data_loader import AAMI_CLASSES


MODEL_NAMES = ["rf", "lgbm", "xgb", "lda", "logreg"]


# -- AAMI cost matrix ---------------------------------------------------------
# Symmetric clinical weighting: missing V is the most expensive error.
# (V-miss > S-miss > F-miss > Q-miss >> N-miss). Used to derive sample_weight
# for cost-sensitive boosters that do not accept class_weight natively.
CLINICAL_WEIGHT = {"N": 1.0, "S": 3.0, "V": 4.0, "F": 2.0, "Q": 1.0}


def clinical_sample_weight(y) -> np.ndarray:
    return np.asarray([CLINICAL_WEIGHT.get(str(lbl), 1.0) for lbl in y],
                      dtype=np.float32)


def balanced_sample_weight(y) -> np.ndarray:
    """1 / freq per class (normalized). Useful when the resampler is not used."""
    counts = Counter(y)
    n = len(y)
    k = len(counts)
    w = {c: n / (k * counts[c]) for c in counts}
    return np.asarray([w[c] for c in y], dtype=np.float32)


# -- Model constructors -------------------------------------------------------
def _rf(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
        class_weight="balanced_subsample", **_):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight=class_weight,
    )


def _lgbm(n_estimators=500, max_depth=-1, learning_rate=0.05, num_leaves=63,
          random_state=42, class_weight="balanced", **_):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=-1,
    )


def _xgb(n_estimators=500, max_depth=6, learning_rate=0.08,
         random_state=42, **_):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )


def _lda(**_):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _logreg(C=1.0, random_state=42, **_):
    return LogisticRegression(
        C=C, max_iter=2000, multi_class="multinomial",
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )


_BUILDERS = {
    "rf": _rf,
    "lgbm": _lgbm,
    "xgb": _xgb,
    "lda": _lda,
    "logreg": _logreg,
}


def build_model(name: str, **kwargs) -> Any:
    name = name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"unknown model '{name}', expected one of {MODEL_NAMES}")
    return _BUILDERS[name](**kwargs)


# -- Fit / predict wrappers that tolerate XGBoost label quirks ----------------
class LabelSafeXGB:
    """Wrap XGBClassifier so it accepts string AAMI labels.

    XGBoost 2.x requires integer targets — we encode once on fit, decode
    once on predict. Exposes a Scikit-learn-compatible surface.
    """

    def __init__(self, clf):
        self.clf = clf
        self.le = LabelEncoder()

    def fit(self, X, y, sample_weight=None):
        y_enc = self.le.fit_transform(y)
        self.clf.fit(X, y_enc, sample_weight=sample_weight)
        # expose classes_ in original label space (str)
        self.classes_ = self.le.classes_
        return self

    def predict(self, X):
        y_enc = self.clf.predict(X)
        return self.le.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_


def fit_with_weights(model, X, y, weighting: str | None = "clinical"):
    """Fit wrapping in sample_weight if the model is a booster; passthrough otherwise.

    weighting ∈ {None, "balanced", "clinical"}.
    """
    sw = None
    if weighting == "balanced":
        sw = balanced_sample_weight(y)
    elif weighting == "clinical":
        sw = clinical_sample_weight(y)

    # XGBoost needs numeric labels
    import xgboost as xgb
    if isinstance(model, xgb.XGBClassifier):
        model = LabelSafeXGB(model)
        model.fit(X, y, sample_weight=sw)
        return model

    try:
        model.fit(X, y, sample_weight=sw)
    except TypeError:
        model.fit(X, y)
    return model
