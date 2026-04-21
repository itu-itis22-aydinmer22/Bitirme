"""Resampling strategies for the AAMI imbalanced problem.

Uniform interface:  ``resample(X, y, strategy, random_state=42)``
strategies:
    - "none"                 — passthrough
    - "smote"                — vanilla SMOTE
    - "borderline"           — BorderlineSMOTE (focus on boundary)
    - "adasyn"               — adaptive synthetic (density-driven)
    - "smotetomek"           — SMOTE + Tomek link cleaning
    - "smoteenn"             — SMOTE + edited nearest-neighbour cleaning
    - "undersample"          — random under-sampling of majority to median

All strategies adapt ``k_neighbors`` when a class is smaller than the default.
"""
from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler


def _min_class(y):
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _drop_singletons(X, y, min_required=2):
    """Drop classes with fewer than ``min_required`` samples (can't synthesize)."""
    uniq, counts = np.unique(y, return_counts=True)
    keep = uniq[counts >= min_required]
    mask = np.isin(y, keep)
    return X[mask], y[mask]


def resample(X, y, strategy: str = "smote", random_state: int = 42,
             default_k: int = 5):
    s = strategy.lower()
    if s == "none" or s is None:
        return X, y

    X, y = _drop_singletons(X, y, min_required=2)
    k = max(1, min(default_k, _min_class(y) - 1))

    if s == "smote":
        return SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
    if s == "borderline":
        return BorderlineSMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
    if s == "adasyn":
        # ADASYN is sensitive to very small classes — guard against failure
        try:
            return ADASYN(random_state=random_state,
                          n_neighbors=k).fit_resample(X, y)
        except (ValueError, RuntimeError):
            # Fall back to vanilla SMOTE if ADASYN can't find neighbors
            return SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
    if s == "smotetomek":
        return SMOTETomek(random_state=random_state,
                          smote=SMOTE(random_state=random_state,
                                      k_neighbors=k)).fit_resample(X, y)
    if s == "smoteenn":
        return SMOTEENN(random_state=random_state,
                        smote=SMOTE(random_state=random_state,
                                    k_neighbors=k)).fit_resample(X, y)
    if s == "undersample":
        # Cap majority class at median count
        uniq, counts = np.unique(y, return_counts=True)
        median = int(np.median(counts))
        strat = {c: min(n, max(median, 2)) for c, n in zip(uniq, counts)}
        return RandomUnderSampler(random_state=random_state,
                                   sampling_strategy=strat).fit_resample(X, y)
    raise ValueError(f"unknown resampling strategy '{strategy}'")
