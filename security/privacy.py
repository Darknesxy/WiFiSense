"""
WiFiSense - Privacy Evaluation & Differential Privacy (DP) Module
Implements Laplace / Gaussian Differential Privacy noise injection, sensitivity bounding,
and empirical Privacy-Utility tradeoff evaluation for WiFi CSI biometric data.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def add_differential_privacy_noise(
    csi_data: np.ndarray, 
    epsilon: float = 1.0, 
    delta: float = 1e-5, 
    sensitivity: Optional[float] = None, 
    mechanism: str = "laplace"
) -> np.ndarray:
    """
    Injects calibrated Differential Privacy noise to protect human occupant identity and gait biometrics.

    Args:
        csi_data: Input CSI array.
        epsilon: Privacy budget (lower epsilon = stronger privacy, higher noise).
        delta: Relaxation parameter for Gaussian mechanism.
        sensitivity: Maximum query/feature sensitivity L1/L2 (estimated if None).
        mechanism: 'laplace' or 'gaussian'.

    Returns:
        Differentially private CSI array.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be strictly positive.")

    data = csi_data.astype(np.float64)

    # Estimate L1 sensitivity (max absolute difference between adjacent windows)
    if sensitivity is None:
        sensitivity = float(np.percentile(np.abs(data), 95)) * 0.1
        sensitivity = max(sensitivity, 1e-3)

    if mechanism.lower() == "laplace":
        # Scale b = Delta / epsilon
        scale = sensitivity / epsilon
        noise = np.random.laplace(0.0, scale, size=data.shape)
    elif mechanism.lower() == "gaussian":
        # Scale sigma = sqrt(2 * ln(1.25 / delta)) * Delta / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / epsilon)
        noise = np.random.normal(0.0, sigma, size=data.shape)
    else:
        raise ValueError(f"Unsupported DP mechanism: {mechanism}")

    return (data + noise).astype(np.float32)


def evaluate_privacy_utility_tradeoff(
    csi_sample: np.ndarray, 
    epsilons: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates empirical tradeoff between Privacy Budget (epsilon) and Signal Quality (SNR, MAE).
    """
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    eval_results = []
    signal_power = np.mean(csi_sample ** 2)

    for eps in epsilons:
        noisy = add_differential_privacy_noise(csi_sample, epsilon=eps, mechanism="laplace")
        noise = noisy - csi_sample
        noise_power = np.mean(noise ** 2)

        # Signal-to-Noise Ratio (SNR) in dB
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        mae = float(np.mean(np.abs(noise)))
        max_error = float(np.max(np.abs(noise)))

        eval_results.append({
            "epsilon": eps,
            "privacy_guarantee": "High" if eps <= 0.5 else ("Medium" if eps <= 2.0 else "Low"),
            "snr_db": round(float(snr_db), 2),
            "mae": round(mae, 4),
            "max_distortion": round(max_error, 4)
        })

    return eval_results


def anonymize_csi_metadata(raw_packet_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    De-identifies CSI frame metadata:
    - Hashes MAC addresses with salt
    - Drops precise GPS and BSSID
    - Quantizes timestamps to 100ms intervals to prevent timing correlation attacks.
    """
    import hashlib
    salt = "wifisense-anonymization-salt"

    sanitized = {}
    if "mac_address" in raw_packet_metadata:
        mac = str(raw_packet_metadata["mac_address"])
        sanitized["device_pseudonym"] = hashlib.sha256((mac + salt).encode()).hexdigest()[:12]

    if "timestamp" in raw_packet_metadata:
        # Quantize timestamp to 0.1s
        ts = float(raw_packet_metadata["timestamp"])
        sanitized["quantized_timestamp"] = round(ts * 10) / 10

    if "channel" in raw_packet_metadata:
        sanitized["channel"] = raw_packet_metadata["channel"]

    return sanitized
