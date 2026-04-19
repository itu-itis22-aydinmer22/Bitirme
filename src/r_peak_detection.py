"""WP1 - Pan-Tompkins QRS / R-peak detector (Pan & Tompkins, IEEE TBME 1985)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _bandpass(signal, fs, low=5.0, high=15.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def pan_tompkins(signal, fs=360, refractory_ms=200):
    """Returns sample indices of detected R-peaks.

    Stages: bandpass (5-15 Hz) -> derivative -> squaring ->
    moving-window integration -> adaptive thresholding.
    """
    filtered = _bandpass(signal, fs)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    win = max(1, int(0.150 * fs))  # 150 ms integration window
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    refractory = int(refractory_ms * 1e-3 * fs)
    peaks, _ = find_peaks(integrated, distance=refractory)
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Adaptive thresholding (simplified Pan-Tompkins)
    spki = np.mean(integrated[peaks][: min(8, len(peaks))])
    npki = np.mean(integrated)
    threshold = npki + 0.25 * (spki - npki)
    accepted = []
    for p in peaks:
        amp = integrated[p]
        if amp > threshold:
            accepted.append(p)
            spki = 0.125 * amp + 0.875 * spki
        else:
            npki = 0.125 * amp + 0.875 * npki
        threshold = npki + 0.25 * (spki - npki)

    # Snap to local maxima of the original (denoised) signal in a +/-50 ms window
    half = int(0.05 * fs)
    refined = []
    for p in accepted:
        lo, hi = max(0, p - half), min(len(signal), p + half)
        refined.append(lo + int(np.argmax(signal[lo:hi])))
    return np.array(sorted(set(refined)), dtype=int)
