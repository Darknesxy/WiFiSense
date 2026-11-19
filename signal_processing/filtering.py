"""
WiFiSense - Signal Filtering Module
Digital filters for WiFi Channel State Information (CSI) denoising and motion band extraction.
"""

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def butterworth_lowpass_filter(
    data: np.ndarray, 
    cutoff: float = 20.0, 
    fs: float = 100.0, 
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase forward-backward Butterworth low-pass filter along axis 0.

    Args:
        data: Array of shape (time_steps, n_subcarriers) or (time_steps,).
        cutoff: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered array of identical shape.
    """
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    is_1d = data.ndim == 1
    arr = data[:, np.newaxis] if is_1d else data
    filtered = filtfilt(b, a, arr, axis=0)
    return filtered[:, 0] if is_1d else filtered


def butterworth_bandpass_filter(
    data: np.ndarray, 
    lowcut: float = 2.0, 
    highcut: float = 35.0, 
    fs: float = 100.0, 
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth band-pass filter isolating human motion Doppler perturbations.

    Args:
        data: Array of shape (time_steps, n_subcarriers) or (time_steps,).
        lowcut: Lower cutoff frequency (Hz) - eliminates static environment multi-path drift.
        highcut: Higher cutoff frequency (Hz) - eliminates high-frequency RF noise.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Band-pass filtered array.
    """
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.01)
    high = min(highcut / nyquist, 0.99)
    b, a = butter(order, [low, high], btype="band", analog=False)

    is_1d = data.ndim == 1
    arr = data[:, np.newaxis] if is_1d else data
    filtered = filtfilt(b, a, arr, axis=0)
    return filtered[:, 0] if is_1d else filtered


def savitzky_golay_filter(
    data: np.ndarray, 
    window_length: int = 15, 
    polyorder: int = 3
) -> np.ndarray:
    """
    Smooths CSI amplitude traces using Savitzky-Golay filtering, preserving local peak morphology.

    Args:
        data: Array of shape (time_steps, n_subcarriers) or (time_steps,).
        window_length: Must be odd positive integer <= length of data.
        polyorder: Degree of the polynomial (< window_length).

    Returns:
        Smoothed array.
    """
    n_samples = data.shape[0]
    if window_length >= n_samples:
        window_length = n_samples - 1 if (n_samples - 1) % 2 != 0 else n_samples - 2
        window_length = max(window_length, polyorder + 2)
        if window_length % 2 == 0:
            window_length += 1

    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)


def denoise_csi_multichannel(
    csi_matrix: np.ndarray, 
    fs: float = 100.0, 
    lowcut: float = 1.0, 
    highcut: float = 40.0, 
    smooth: bool = True
) -> np.ndarray:
    """
    Cascaded denoising pipeline:
    1. Bandpass filter for motion frequency extraction (1-40 Hz).
    2. Optional Savitzky-Golay polynomial smoothing.
    """
    bp_data = butterworth_bandpass_filter(csi_matrix, lowcut=lowcut, highcut=highcut, fs=fs)
    if smooth and bp_data.shape[0] > 15:
        return savitzky_golay_filter(bp_data)
    return bp_data
