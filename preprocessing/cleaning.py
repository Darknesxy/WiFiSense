"""
WiFiSense - Signal Cleaning & Sanitization Module
Implements Hampel outlier filtering, CSI phase sanitization, missing packet interpolation, and normalization.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import interpolate


def hampel_filter(data: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Applies the Hampel filter along the time axis (axis 0) to detect and replace outliers using
    Median Absolute Deviation (MAD).

    Args:
        data: Input numpy array of shape (time_steps, subcarriers) or (time_steps,).
        window_size: Half-window size for rolling median.
        n_sigmas: Multiplier for MAD threshold.

    Returns:
        Filtered numpy array with outliers replaced by the window median.
    """
    is_1d = data.ndim == 1
    arr = data[:, np.newaxis] if is_1d else data.copy()
    n_samples, n_sub = arr.shape
    filtered = arr.copy()

    # Scale factor for Gaussian distribution
    k = 1.4826

    for sub_idx in range(n_sub):
        channel = arr[:, sub_idx]
        for t in range(n_samples):
            start = max(0, t - window_size)
            end = min(n_samples, t + window_size + 1)
            window = channel[start:end]
            
            median = np.median(window)
            mad = k * np.median(np.abs(window - median))
            threshold = n_sigmas * mad

            if mad > 1e-8:
                if np.abs(channel[t] - median) > threshold:
                    filtered[t, sub_idx] = median
            else:
                if np.abs(channel[t] - median) > 1e-4:
                    filtered[t, sub_idx] = median

    return filtered[:, 0] if is_1d else filtered


def phase_sanitization(raw_phase: np.ndarray, subcarrier_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Sanitizes raw CSI phase by removing Carrier Frequency Offset (CFO) and 
    Sampling Frequency Offset (SFO) using linear fitting across subcarriers.

    Phase model: phi_measured(i) = phi_true(i) - 2*pi*(k_i/N)*delta_t + beta + noise
    Linear transformation finds slope (a) and intercept (b) to adjust:
        phi_sanitized(i) = phi_measured(i) - a * k_i - b

    Args:
        raw_phase: Phase array of shape (time_steps, n_subcarriers).
        subcarrier_indices: Array of subcarrier index numbers (e.g., -28 to 28 for 56 subcarriers).

    Returns:
        Sanitized phase array of the same shape.
    """
    if raw_phase.ndim == 1:
        raw_phase = raw_phase[np.newaxis, :]
    
    n_samples, n_subcarriers = raw_phase.shape

    if subcarrier_indices is None:
        subcarrier_indices = np.arange(-n_subcarriers // 2, n_subcarriers // 2)

    # First unwrap phase along subcarrier axis
    unwrapped_phase = np.unwrap(raw_phase, axis=1)

    k = subcarrier_indices
    k_mean = np.mean(k)
    k_var = np.var(k)

    sanitized = np.zeros_like(unwrapped_phase)

    for t in range(n_samples):
        y = unwrapped_phase[t, :]
        if k_var > 1e-8:
            slope = np.cov(k, y, bias=True)[0, 1] / k_var
            intercept = np.mean(y) - slope * k_mean
            sanitized[t, :] = y - slope * k - intercept
        else:
            sanitized[t, :] = y - np.mean(y)

    return sanitized


def interpolate_missing_packets(
    timestamps: np.ndarray, 
    csi_matrix: np.ndarray, 
    target_sampling_rate_hz: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates irregular or missing CSI packet arrivals onto a uniform temporal grid.

    Args:
        timestamps: 1D array of packet reception times in seconds.
        csi_matrix: Array of shape (n_packets, n_subcarriers).
        target_sampling_rate_hz: Desired constant sampling rate.

    Returns:
        uniform_timestamps: 1D array of uniform timestamps.
        uniform_csi: Interpolated CSI array of shape (target_samples, n_subcarriers).
    """
    if len(timestamps) < 2:
        return timestamps, csi_matrix

    t_start, t_end = timestamps[0], timestamps[-1]
    step = 1.0 / target_sampling_rate_hz
    uniform_timestamps = np.arange(t_start, t_end, step)

    if len(uniform_timestamps) == 0:
        return timestamps, csi_matrix

    # Linear interpolation along each subcarrier
    f_interp = interpolate.interp1d(
        timestamps, 
        csi_matrix, 
        axis=0, 
        kind="linear", 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    uniform_csi = f_interp(uniform_timestamps)
    return uniform_timestamps, uniform_csi


def normalize_csi(csi_data: np.ndarray, method: str = "z_score") -> np.ndarray:
    """
    Normalizes CSI matrix.

    Args:
        csi_data: Array of shape (time_steps, n_subcarriers).
        method: "z_score" (zero mean, unit variance) or "min_max" (0 to 1).

    Returns:
        Normalized array of identical shape.
    """
    data = csi_data.astype(np.float32)
    if method == "z_score":
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    elif method == "min_max":
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        diff = max_val - min_val
        diff[diff == 0] = 1.0
        return (data - min_val) / diff
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_csi_packet_stream(
    raw_amplitude: np.ndarray, 
    raw_phase: Optional[np.ndarray] = None,
    window_size: int = 5,
    normalize: bool = True
) -> np.ndarray:
    """
    End-to-end cleaning pipeline for raw incoming WiFi CSI data.

    Steps:
        1. Hampel outlier filtering on amplitude.
        2. Phase sanitization (if phase provided).
        3. Standard normalization.

    Returns:
        Cleaned CSI representation (amplitude matrix or sanitized representation).
    """
    cleaned_amp = hampel_filter(raw_amplitude, window_size=window_size)
    if normalize:
        cleaned_amp = normalize_csi(cleaned_amp, method="z_score")

    if raw_phase is not None:
        sanitized_phase = phase_sanitization(raw_phase)
        return cleaned_amp, sanitized_phase
    
    return cleaned_amp
