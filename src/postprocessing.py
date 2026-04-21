"""Sequence-level smoothing for beat-level classifier output.

The raw classifier emits per-beat predictions.  Because arrhythmias almost
always appear in short *runs* (ventricular couplets, bigeminy), fitting a
per-patient Hidden Markov Model on top of the classifier's probabilities
cleans up isolated flip-flops and nudges the decision toward a clinically
plausible sequence.

We use a lightweight HMM with observation probabilities taken directly from
the classifier (P(obs=ĉ | state=c) ≈ classifier confidence) and a soft
transition matrix estimated on the training data, plus a scalar prior on
self-transition to reward clinical inertia.
"""
from __future__ import annotations

import numpy as np

from .data_loader import AAMI_CLASSES


def estimate_transition_matrix(y_seq, groups_seq, classes=AAMI_CLASSES,
                                self_prior: float = 0.8,
                                smoothing: float = 1.0) -> np.ndarray:
    """Patient-aware transition counts: P(state_t+1 | state_t).

    ``self_prior`` blends the empirical matrix with an identity prior so an
    under-sampled class (F, Q) still has sane self-transitions.
    """
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    T = np.full((n, n), smoothing, dtype=np.float64)
    y_seq = np.asarray(y_seq)
    groups_seq = np.asarray(groups_seq)
    for g in np.unique(groups_seq):
        mask = groups_seq == g
        ys = y_seq[mask]
        if len(ys) < 2:
            continue
        for a, b in zip(ys[:-1], ys[1:]):
            if a in idx and b in idx:
                T[idx[a], idx[b]] += 1
    T = T / T.sum(axis=1, keepdims=True)
    T = self_prior * np.eye(n) + (1 - self_prior) * T
    T = T / T.sum(axis=1, keepdims=True)
    return T


def viterbi_smooth(proba, groups, transitions, classes=AAMI_CLASSES):
    """Run Viterbi per-patient; return smoothed labels.

    ``proba`` shape (N, K) — classifier posteriors, columns aligned to ``classes``.
    ``groups`` shape (N,)  — patient id, beats are assumed ordered within patient.
    """
    proba = np.asarray(proba, dtype=np.float64)
    groups = np.asarray(groups)
    N, K = proba.shape
    log_emit = np.log(np.clip(proba, 1e-12, 1.0))
    log_trans = np.log(np.clip(transitions, 1e-12, 1.0))
    out = np.empty(N, dtype=object)
    log_pi = np.log(np.ones(K) / K)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        T = len(idx)
        V = np.empty((T, K))
        back = np.empty((T, K), dtype=np.int32)
        V[0] = log_pi + log_emit[idx[0]]
        back[0] = 0
        for t in range(1, T):
            step = V[t - 1][:, None] + log_trans + log_emit[idx[t]][None, :]
            back[t] = step.argmax(axis=0)
            V[t] = step.max(axis=0)
        best = np.empty(T, dtype=np.int32)
        best[-1] = V[-1].argmax()
        for t in range(T - 2, -1, -1):
            best[t] = back[t + 1, best[t + 1]]
        for local, global_i in enumerate(idx):
            out[global_i] = classes[best[local]]
    return np.asarray(out, dtype=object).astype(str)
