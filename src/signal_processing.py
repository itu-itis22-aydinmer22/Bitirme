"""WP1 - Signal pre-processing: bandpass filter + Daubechies db4 wavelet denoising."""
import numpy as np
import pywt
from scipy.signal import butter, filtfilt


FS_DEFAULT = 360  # MIT-BIH sampling rate


def bandpass_filter(signal, fs=FS_DEFAULT, low=0.5, high=40.0, order=4):
    """Zero-phase Butterworth band-pass filter (0.5-40 Hz removes baseline wander
    and high-frequency noise while preserving QRS morphology)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def wavelet_denoise(signal, wavelet="db4", level=None, mode="soft"):
    """DWT denoising with Daubechies db4. Universal threshold derived from the
    finest detail scale via the median-absolute-deviation noise estimator."""
    if level is None:
        level = min(pywt.dwt_max_level(len(signal), wavelet), 8)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=threshold, mode=mode) for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)]


def preprocess(signal, fs=FS_DEFAULT):
    """Full preprocessing chain used by the pipeline."""
    return wavelet_denoise(bandpass_filter(signal, fs=fs))


def add_awgn(signal, snr_db):
    """Inject additive white Gaussian noise at a target SNR (dB) for the
    robustness experiment described in the report."""
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise
