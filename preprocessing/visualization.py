"""
WiFiSense - Signal Visualization Module
Provides plotting utilities for CSI heatmaps, subcarrier dynamics, phase profiles, and spectrograms.
"""

from typing import Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plot_csi_amplitude_heatmap(
    csi_matrix: np.ndarray, 
    title: str = "WiFi CSI Amplitude Heatmap (Subcarriers vs Time)",
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Generates a 2D Heatmap of CSI Amplitude over time and subcarriers.

    Args:
        csi_matrix: Array of shape (time_steps, n_subcarriers).
        title: Plot title.
        cmap: Matplotlib colormap.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(csi_matrix.T, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Packet / Time Index", fontsize=10)
    ax.set_ylabel("Subcarrier Index", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CSI Amplitude", fontsize=10)
    fig.tight_layout()
    return fig


def plot_subcarrier_streams(
    csi_matrix: np.ndarray, 
    subcarrier_indices: Optional[List[int]] = None,
    title: str = "CSI Amplitude Variations Across Selected Subcarriers"
) -> plt.Figure:
    """
    Plots temporal amplitude traces for selected subcarrier indices.

    Args:
        csi_matrix: Array of shape (time_steps, n_subcarriers).
        subcarrier_indices: Indices of subcarriers to plot (default: 5 representative channels).
        title: Plot title.

    Returns:
        matplotlib Figure object.
    """
    n_time, n_sub = csi_matrix.shape
    if subcarrier_indices is None:
        subcarrier_indices = list(np.linspace(0, n_sub - 1, min(5, n_sub), dtype=int))

    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in subcarrier_indices:
        ax.plot(csi_matrix[:, idx], label=f"Subcarrier #{idx}", alpha=0.85)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step (Packets)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_phase_distribution(
    raw_phase: np.ndarray, 
    sanitized_phase: np.ndarray, 
    sample_index: int = 0
) -> plt.Figure:
    """
    Compares raw vs sanitized CSI phase profiles for a specific time packet.

    Args:
        raw_phase: Array of shape (time_steps, n_subcarriers).
        sanitized_phase: Array of shape (time_steps, n_subcarriers).
        sample_index: Packet index to inspect.

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(raw_phase[sample_index], marker="o", color="crimson", linewidth=1.5, markersize=3)
    ax1.set_title(f"Raw CSI Phase (Packet #{sample_index})", fontsize=11)
    ax1.set_xlabel("Subcarrier Index")
    ax1.set_ylabel("Phase (radians)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(sanitized_phase[sample_index], marker="s", color="royalblue", linewidth=1.5, markersize=3)
    ax2.set_title(f"Sanitized CSI Phase (Packet #{sample_index})", fontsize=11)
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("Sanitized Phase (radians)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("CSI Phase Linear Transformation & Sanitization", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def generate_spectrogram(
    signal_series: np.ndarray, 
    fs: float = 100.0, 
    nperseg: int = 64, 
    noverlap: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Figure]:
    """
    Computes and plots Doppler Spectrogram (STFT) for a CSI signal.

    Args:
        signal_series: 1D array representing principal component or representative subcarrier.
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment for STFT.
        noverlap: Number of points of overlap between segments.

    Returns:
        (frequencies, times, Sxx, fig)
    """
    f, t, sxx = spectrogram(signal_series, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-10), shading="gouraud", cmap="inferno")
    ax.set_title("Doppler Spectrogram (STFT) of CSI Perturbation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency [Hz]", fontsize=10)
    ax.set_xlabel("Time [sec]", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power Spectral Density [dB/Hz]", fontsize=10)
    fig.tight_layout()
    return f, t, sxx, fig
