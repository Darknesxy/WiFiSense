"""
WiFiSense Signal Processing Package
Provides filtering (Butterworth, Savitzky-Golay) and feature extraction (STFT, Doppler, statistical features, PCA).
"""

from .filtering import (
    butterworth_lowpass_filter,
    butterworth_bandpass_filter,
    savitzky_golay_filter,
    denoise_csi_multichannel
)
from .feature_extraction import (
    extract_statistical_features,
    extract_doppler_features,
    extract_pca_features,
    extract_full_feature_vector
)

__all__ = [
    "butterworth_lowpass_filter",
    "butterworth_bandpass_filter",
    "savitzky_golay_filter",
    "denoise_csi_multichannel",
    "extract_statistical_features",
    "extract_doppler_features",
    "extract_pca_features",
    "extract_full_feature_vector"
]
