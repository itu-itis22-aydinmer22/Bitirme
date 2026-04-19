"""WP2 - Hybrid feature extraction (temporal + amplitude + PCA morphology).

The raw-signal implementation here mirrors the feature schema present in the
preprocessed CSVs we train on, so the same Random Forest can be deployed on
either representation.
"""
import numpy as np
from sklearn.decomposition import PCA


QRS_WINDOW = 0.10   # +/- 100 ms around R-peak for QRS morphology
P_WINDOW = (0.20, 0.05)  # search P-wave 50-200 ms before R
T_WINDOW = (0.10, 0.40)  # search T-wave 100-400 ms after R
N_PCA = 5


def _safe_argmax(segment, default=0):
    return int(np.argmax(segment)) if len(segment) else default


def _qrs_segment(signal, r, fs):
    half = int(QRS_WINDOW * fs)
    lo, hi = max(0, r - half), min(len(signal), r + half)
    return signal[lo:hi], lo


def _peak_amplitudes(signal, r, fs):
    qrs, lo = _qrs_segment(signal, r, fs)
    if len(qrs) == 0:
        return dict.fromkeys("PQRST", 0.0)

    r_amp = signal[r]
    q_idx = lo + _safe_argmax(-qrs[: r - lo])
    s_off = _safe_argmax(-signal[r:min(len(signal), r + int(QRS_WINDOW * fs))])
    s_idx = r + s_off

    p_lo = max(0, r - int(P_WINDOW[0] * fs))
    p_hi = max(0, r - int(P_WINDOW[1] * fs))
    p_idx = p_lo + _safe_argmax(signal[p_lo:p_hi])

    t_lo = min(len(signal) - 1, r + int(T_WINDOW[0] * fs))
    t_hi = min(len(signal), r + int(T_WINDOW[1] * fs))
    t_idx = t_lo + _safe_argmax(signal[t_lo:t_hi])

    return {
        "P": float(signal[p_idx]),
        "Q": float(signal[q_idx]),
        "R": float(r_amp),
        "S": float(signal[s_idx]),
        "T": float(signal[t_idx]),
        "_idx": (p_idx, q_idx, r, s_idx, t_idx),
    }


def temporal_features(r_peaks, fs):
    """RR intervals (in samples) for every beat."""
    rr = np.diff(r_peaks) / fs
    pre_rr = np.concatenate([[rr[0]], rr]) if len(rr) else np.array([])
    post_rr = np.concatenate([rr, [rr[-1]]]) if len(rr) else np.array([])
    return pre_rr, post_rr


def morphology_pca(qrs_matrix, n_components=N_PCA):
    """Train a PCA on aligned QRS complexes and return the projection."""
    pca = PCA(n_components=n_components)
    return pca, pca.fit_transform(qrs_matrix)


def extract_features(signal, r_peaks, fs=360, pca_model=None):
    """Build hybrid feature vectors for all detected beats."""
    if len(r_peaks) < 2:
        return np.zeros((0, 10 + N_PCA))

    pre_rr, post_rr = temporal_features(r_peaks, fs)
    rows, qrs_matrix = [], []
    half = int(QRS_WINDOW * fs)

    for r in r_peaks:
        qrs, lo = _qrs_segment(signal, r, fs)
        if len(qrs) < 2 * half:
            qrs = np.pad(qrs, (0, 2 * half - len(qrs)))
        qrs_matrix.append(qrs[: 2 * half])

    qrs_matrix = np.vstack(qrs_matrix)
    if pca_model is None:
        pca_model, morph = morphology_pca(qrs_matrix)
    else:
        morph = pca_model.transform(qrs_matrix)

    for i, r in enumerate(r_peaks):
        amps = _peak_amplitudes(signal, r, fs)
        p_i, q_i, _, s_i, t_i = amps["_idx"]
        qrs_int = (s_i - q_i) / fs
        pq_int = (q_i - p_i) / fs
        qt_int = (t_i - q_i) / fs
        st_int = (t_i - s_i) / fs
        row = [
            pre_rr[i], post_rr[i],
            amps["P"], amps["T"], amps["R"], amps["S"], amps["Q"],
            qrs_int, pq_int, qt_int, st_int,
            *morph[i].tolist(),
        ]
        rows.append(row)
    return np.asarray(rows), pca_model
