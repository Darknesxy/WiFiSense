import pytest
import numpy as np
from preprocessing.cleaning import (
    hampel_filter,
    phase_sanitization,
    interpolate_missing_packets,
    normalize_csi,
    preprocess_csi_packet_stream
)


def test_hampel_filter():
    # Signal with artificial outlier spike
    data = np.ones((50, 4))
    data[20, 1] = 50.0  # Massive outlier

    cleaned = hampel_filter(data, window_size=5, n_sigmas=3.0)
    assert cleaned.shape == (50, 4)
    assert cleaned[20, 1] < 10.0  # Outlier replaced by median
    assert np.allclose(cleaned[0:15, 1], 1.0)


def test_phase_sanitization():
    n_samples, n_sub = 30, 56
    sub_indices = np.arange(-28, 28)
    # Synthetic linear phase slope
    raw_phase = np.zeros((n_samples, n_sub))
    for t in range(n_samples):
        raw_phase[t] = 0.5 * sub_indices + 1.2 + np.random.normal(0, 0.01, n_sub)

    sanitized = phase_sanitization(raw_phase, sub_indices)
    assert sanitized.shape == (n_samples, n_sub)
    # Slope should be removed, variance around zero
    assert np.mean(np.abs(sanitized)) < 0.5


def test_interpolate_missing_packets():
    # Non-uniform timestamps with a gap
    timestamps = np.array([0.0, 0.01, 0.02, 0.05, 0.06])
    csi_data = np.random.randn(5, 10)

    uniform_t, uniform_csi = interpolate_missing_packets(timestamps, csi_data, target_sampling_rate_hz=100.0)
    assert len(uniform_t) > len(timestamps)
    assert uniform_csi.shape[1] == 10


def test_normalize_csi():
    data = np.random.normal(10.0, 2.0, (40, 8))
    z_norm = normalize_csi(data, method="z_score")
    assert np.allclose(np.mean(z_norm, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(z_norm, axis=0), 1.0, atol=1e-5)

    min_max = normalize_csi(data, method="min_max")
    assert np.all(min_max >= 0.0)
    assert np.all(min_max <= 1.0 + 1e-6)
