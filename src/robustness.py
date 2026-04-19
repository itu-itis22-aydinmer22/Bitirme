"""Robustness evaluation - inject AWGN at controlled SNR levels.

Because the published CSV is already feature-extracted (the raw waveform is
gone), we mimic real-world sensor noise by perturbing the engineered features
with calibrated Gaussian noise. The perturbation is scaled per-feature using
the training-set standard deviation so the requested SNR translates to the
same per-feature signal-to-noise ratio it would in the time-domain signal.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data_loader import AAMI_CLASSES


def inject_feature_noise(X, snr_db, rng=None):
    rng = np.random.default_rng(rng)
    sigma_signal = X.std(axis=0)
    noise_std = sigma_signal / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_std, size=X.shape).astype(X.dtype)
    return X + noise


def evaluate_under_noise(model, scaler, X, y, snr_levels=(20, 15, 10, 5, 0), seed=0):
    """Returns a dict {snr_db: {accuracy, macro_f1, v_f1}}."""
    out = {}
    rng = np.random.default_rng(seed)
    X_s = scaler.transform(X)
    y_clean = model.predict(X_s)
    out["clean"] = {
        "accuracy": float(accuracy_score(y, y_clean)),
        "macro_f1": float(f1_score(y, y_clean, labels=AAMI_CLASSES, average="macro", zero_division=0)),
        "v_f1": float(f1_score(y, y_clean, labels=["V"], average="macro", zero_division=0)),
    }
    for snr in snr_levels:
        X_noisy = inject_feature_noise(X_s, snr_db=snr, rng=rng)
        y_pred = model.predict(X_noisy)
        out[snr] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "macro_f1": float(f1_score(y, y_pred, labels=AAMI_CLASSES, average="macro", zero_division=0)),
            "v_f1": float(f1_score(y, y_pred, labels=["V"], average="macro", zero_division=0)),
        }
    return out
