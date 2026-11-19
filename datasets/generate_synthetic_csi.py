"""
WiFiSense - Synthetic CSI Dataset Generator
Generates realistic WiFi Channel State Information (CSI) packets mimicking 802.11n/ac 
subcarriers across standard Human Activity Recognition (HAR) scenarios.
"""

import os
from typing import Tuple, Dict, List
import numpy as np


ACTIVITIES = ["Empty_Room", "Walking", "Sitting_Down", "Standing_Up", "Falling", "Gesturing"]


def generate_synthetic_activity_sample(
    activity: str, 
    n_packets: int = 200, 
    n_subcarriers: int = 56, 
    fs: float = 100.0,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single (amplitude, phase) CSI window for a specified activity.

    Args:
        activity: One of ACTIVITIES.
        n_packets: Time steps per window.
        n_subcarriers: Number of OFDM subcarriers (e.g. 56 or 64).
        fs: Sampling rate in Hz.
        random_state: Optional seed for reproducibility.

    Returns:
        amplitude: Array of shape (n_packets, n_subcarriers).
        phase: Array of shape (n_packets, n_subcarriers).
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, n_packets / fs, n_packets)

    # Base channel profile (frequency-selective fading)
    subcarrier_decay = np.exp(-np.linspace(0, 1.5, n_subcarriers))
    base_amp = 25.0 * subcarrier_decay[np.newaxis, :]
    noise_amp = rng.normal(0, 0.4, size=(n_packets, n_subcarriers))

    # Static phase profile with random initial offset
    subcarrier_idx = np.arange(-n_subcarriers // 2, n_subcarriers // 2)
    sfo_drift = 0.08 * subcarrier_idx
    cfo_drift = rng.uniform(-1.0, 1.0)
    raw_phase = np.tile(sfo_drift + cfo_drift, (n_packets, 1)) + rng.normal(0, 0.15, (n_packets, n_subcarriers))

    motion_amp = np.zeros((n_packets, n_subcarriers))

    if activity == "Empty_Room":
        # Static environmental multipath, only subtle noise variations
        pass

    elif activity == "Walking":
        # Periodic Doppler fluctuations (gait cadence ~1.8 Hz + harmonics)
        gait_freq = rng.uniform(1.6, 2.2)
        stride_effect = np.sin(2 * np.pi * gait_freq * t)[:, np.newaxis]
        subcarrier_modulation = np.cos(np.linspace(0, 3 * np.pi, n_subcarriers))[np.newaxis, :]
        motion_amp = 6.0 * stride_effect * subcarrier_modulation

    elif activity == "Sitting_Down":
        # Smooth downward transition and settling
        decay = np.exp(-((t - 1.0) ** 2) / 0.3)[:, np.newaxis]
        motion_amp = -4.5 * decay * (1.0 + 0.3 * np.sin(2 * np.pi * 3.0 * t)[:, np.newaxis])

    elif activity == "Standing_Up":
        # Upward dynamic transition
        rise = np.exp(-((t - 0.8) ** 2) / 0.25)[:, np.newaxis]
        motion_amp = 5.0 * rise * (1.0 + 0.2 * np.cos(2 * np.pi * 3.5 * t)[:, np.newaxis])

    elif activity == "Falling":
        # Violent acceleration peak followed by sudden stillness (fall event at t=0.7s)
        fall_pulse = 12.0 * np.exp(-((t - 0.7) ** 2) / 0.04)[:, np.newaxis]
        rebound = -3.5 * np.exp(-((t - 1.0) ** 2) / 0.08)[:, np.newaxis]
        motion_amp = fall_pulse + rebound

    elif activity == "Gesturing":
        # High-frequency, localized hand oscillations (~8 Hz)
        gesture_freq = rng.uniform(6.0, 9.0)
        hand_motion = np.sin(2 * np.pi * gesture_freq * t)[:, np.newaxis]
        # Active primarily on subset of upper subcarriers
        sub_mask = np.linspace(0.2, 1.5, n_subcarriers)[np.newaxis, :]
        motion_amp = 3.5 * hand_motion * sub_mask

    amplitude = np.maximum(base_amp + motion_amp + noise_amp, 0.1)
    phase = raw_phase + 0.1 * motion_amp

    return amplitude.astype(np.float32), phase.astype(np.float32)


def generate_csi_dataset(
    samples_per_class: int = 50,
    n_packets: int = 200,
    n_subcarriers: int = 56,
    fs: float = 100.0,
    output_dir: str = "datasets"
) -> Dict[str, Any]:
    """
    Generates and saves a full dataset of CSI samples across all activities.

    Returns:
        Dict containing file paths, X_amp, X_phase, y, and label_map.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_amps = []
    all_phases = []
    all_labels = []

    label_to_idx = {name: idx for idx, name in enumerate(ACTIVITIES)}

    total_samples = samples_per_class * len(ACTIVITIES)
    for class_idx, act in enumerate(ACTIVITIES):
        for s in range(samples_per_class):
            seed = class_idx * 1000 + s
            amp, phase = generate_synthetic_activity_sample(
                act, n_packets=n_packets, n_subcarriers=n_subcarriers, fs=fs, random_state=seed
            )
            all_amps.append(amp)
            all_phases.append(phase)
            all_labels.append(class_idx)

    X_amp = np.stack(all_amps, axis=0)       # Shape: (N, time_steps, n_subcarriers)
    X_phase = np.stack(all_phases, axis=0)   # Shape: (N, time_steps, n_subcarriers)
    y = np.array(all_labels, dtype=np.int64) # Shape: (N,)

    out_file = os.path.join(output_dir, "wifi_csi_dataset.npz")
    np.savez_compressed(
        out_file,
        X_amp=X_amp,
        X_phase=X_phase,
        y=y,
        activities=ACTIVITIES,
        fs=fs
    )

    print(f"[Dataset] Generated {total_samples} samples across {len(ACTIVITIES)} classes.")
    print(f"[Dataset] Saved to {out_file} (Amplitude: {X_amp.shape}, Labels: {y.shape})")

    return {
        "file_path": out_file,
        "X_amp": X_amp,
        "X_phase": X_phase,
        "y": y,
        "activities": ACTIVITIES
    }


if __name__ == "__main__":
    generate_csi_dataset(samples_per_class=40)
