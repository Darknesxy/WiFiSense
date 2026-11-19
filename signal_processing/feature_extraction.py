"""
WiFiSense - Feature Extraction Module
Extracts statistical, frequency-domain (Doppler), and principal component features from CSI streams.
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA


def extract_statistical_features(csi_matrix: np.ndarray) -> np.ndarray:
    """
    Computes statistical summary features per subcarrier:
    Mean, Std, Skewness, Kurtosis, Min, Max, Range, RMS, Energy.

    Args:
        csi_matrix: Array of shape (time_steps, n_subcarriers).

    Returns:
        1D feature vector of length (9 * n_subcarriers).
    """
    mean_val = np.mean(csi_matrix, axis=0)
    std_val = np.std(csi_matrix, axis=0)
    skew_val = stats.skew(csi_matrix, axis=0, nan_policy="omit")
    kurt_val = stats.kurtosis(csi_matrix, axis=0, nan_policy="omit")
    min_val = np.min(csi_matrix, axis=0)
    max_val = np.max(csi_matrix, axis=0)
    ptp_val = max_val - min_val
    rms_val = np.sqrt(np.mean(csi_matrix**2, axis=0))
    energy_val = np.sum(csi_matrix**2, axis=0)

    # Clean any potential nans
    feats = np.concatenate([
        mean_val, std_val, skew_val, kurt_val, min_val, max_val, ptp_val, rms_val, energy_val
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_doppler_features(csi_matrix: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Computes frequency and Doppler characteristics:
    Dominant frequency, Spectral Centroid, Spectral Spread, Spectral Entropy.

    Args:
        csi_matrix: Array of shape (time_steps, n_subcarriers).
        fs: Sampling frequency in Hz.

    Returns:
        1D feature vector.
    """
    n_samples, n_sub = csi_matrix.shape
    freqs = rfftfreq(n_samples, d=1.0 / fs)
    fft_vals = np.abs(rfft(csi_matrix, axis=0))

    # Power spectral density
    psd = (fft_vals ** 2) / n_samples
    psd_sum = np.sum(psd, axis=0, keepdims=True)
    psd_sum[psd_sum == 0] = 1e-12
    norm_psd = psd / psd_sum

    # Dominant frequency
    dom_freq_idx = np.argmax(psd, axis=0)
    dom_freq = freqs[dom_freq_idx]

    # Spectral centroid: sum(f * P(f)) / sum(P(f))
    spectral_centroid = np.sum(freqs[:, np.newaxis] * norm_psd, axis=0)

    # Spectral spread
    freq_diff_sq = (freqs[:, np.newaxis] - spectral_centroid[np.newaxis, :]) ** 2
    spectral_spread = np.sqrt(np.sum(freq_diff_sq * norm_psd, axis=0))

    # Spectral entropy: -sum(p * log2(p))
    log_p = np.log2(norm_psd + 1e-12)
    spectral_entropy = -np.sum(norm_psd * log_p, axis=0)

    features = np.concatenate([dom_freq, spectral_centroid, spectral_spread, spectral_entropy])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_pca_features(csi_matrix: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """
    Extracts top Principal Components capturing cross-subcarrier correlation.

    Args:
        csi_matrix: Array of shape (time_steps, n_subcarriers).
        n_components: Number of components.

    Returns:
        transformed: Array of shape (time_steps, n_components).
        pca_model: Fitted scikit-learn PCA instance.
    """
    n_components = min(n_components, csi_matrix.shape[1], csi_matrix.shape[0])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(csi_matrix)
    return transformed, pca


def extract_full_feature_vector(csi_matrix: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Assembles a complete hand-crafted feature vector combining statistical and Doppler dynamics.
    Suitable for baseline ML models (Random Forest, SVM, LightGBM).

    Args:
        csi_matrix: Array of shape (time_steps, n_subcarriers).
        fs: Sampling rate.

    Returns:
        1D feature vector.
    """
    stat_feats = extract_statistical_features(csi_matrix)
    dop_feats = extract_doppler_features(csi_matrix, fs=fs)
    return np.concatenate([stat_feats, dop_feats])
