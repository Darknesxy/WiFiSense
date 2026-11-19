import pytest
import numpy as np
from signal_processing.filtering import (
    butterworth_lowpass_filter,
    butterworth_bandpass_filter,
    savitzky_golay_filter
)
from signal_processing.feature_extraction import (
    extract_statistical_features,
    extract_doppler_features,
    extract_pca_features,
    extract_full_feature_vector
)


def test_butterworth_filters():
    t = np.linspace(0, 1.0, 100)
    # 5 Hz motion signal + 45 Hz noise
    sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 45 * t)
    sig_mat = np.tile(sig[:, np.newaxis], (1, 4))

    lp = butterworth_lowpass_filter(sig_mat, cutoff=20.0, fs=100.0)
    assert lp.shape == sig_mat.shape

    bp = butterworth_bandpass_filter(sig_mat, lowcut=2.0, highcut=25.0, fs=100.0)
    assert bp.shape == sig_mat.shape


def test_savitzky_golay():
    data = np.random.randn(50, 6)
    smoothed = savitzky_golay_filter(data, window_length=11, polyorder=2)
    assert smoothed.shape == data.shape


def test_feature_extraction():
    csi_matrix = np.random.randn(100, 10)

    stats = extract_statistical_features(csi_matrix)
    assert len(stats) == 9 * 10
    assert not np.isnan(stats).any()

    doppler = extract_doppler_features(csi_matrix, fs=100.0)
    assert len(doppler) == 4 * 10
    assert not np.isnan(doppler).any()

    full_vec = extract_full_feature_vector(csi_matrix, fs=100.0)
    assert len(full_vec) == (9 * 10) + (4 * 10)

    pca_transformed, pca_model = extract_pca_features(csi_matrix, n_components=3)
    assert pca_transformed.shape == (100, 3)
