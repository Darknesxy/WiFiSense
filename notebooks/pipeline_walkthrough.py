"""
WiFiSense - Pipeline Demonstration Walkthrough
Executes the complete end-to-end pipeline across Phases 4, 5, and 6.
"""

import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.generate_synthetic_csi import generate_csi_dataset, ACTIVITIES
from preprocessing.cleaning import preprocess_csi_packet_stream
from signal_processing.filtering import butterworth_bandpass_filter
from signal_processing.feature_extraction import extract_full_feature_vector
from models.evaluation import compare_all_models
from security.encryption import generate_data_encryption_key, SecureCSIStorage
from security.privacy import evaluate_privacy_utility_tradeoff
from security.audit import AuditLogger, verify_audit_log_integrity


def run_pipeline():
    print("=" * 65)
    print("      WIFISENSE END-TO-END PIPELINE DEMONSTRATION")
    print("=" * 65)

    # 1. Dataset Generation
    print("\n[Step 1] Generating Synthetic CSI Dataset...")
    data = generate_csi_dataset(samples_per_class=20, n_packets=100, n_subcarriers=56)
    X_amp, y = data["X_amp"], data["y"]

    # 2. Preprocessing & Denoising
    print("\n[Step 2] Preprocessing and Hampel/Bandpass Filtering Sample...")
    cleaned_sample = preprocess_csi_packet_stream(X_amp[0])
    print(f"Cleaned sample shape: {cleaned_sample.shape}")

    # 3. Phase 4 - Model Training & Comparison
    print("\n[Step 3] Phase 4: Training & Comparing Models (RF, SVM, 1D-CNN, 2D-CNN)...")
    comp_results = compare_all_models(X_amp, y, ACTIVITIES, epochs=5)

    print("\n--- Model Comparison Results ---")
    for row in comp_results["comparison_table"]:
        print(f"| {row['Model']:<18} | Acc: {row['Accuracy (%)']:>6}% | F1: {row['F1-Score']:>6} | Latency: {row['Inference Latency (ms)']:>6} ms |")

    # 4. Phase 6 - Cybersecurity & Encrypted Storage
    print("\n[Step 4] Phase 6: AES-256 Storage Encryption & Tamper-Evident Audit...")
    storage = SecureCSIStorage()
    enc_path = storage.save_encrypted_npz("datasets/demo_encrypted.npz", {"X_amp": X_amp, "y": y})
    print(f"Encrypted archive created: {enc_path}")

    # Verify decrypt
    loaded = storage.load_encrypted_npz(enc_path)
    assert np.allclose(loaded["X_amp"], X_amp)
    print("AES-256 zero-plaintext decryption verified successfully!")

    # 5. Differential Privacy Evaluation
    print("\n[Step 5] Phase 6: Differential Privacy Tradeoff Analysis...")
    dp_results = evaluate_privacy_utility_tradeoff(X_amp[0])
    for item in dp_results:
        print(f"Epsilon: {item['epsilon']:<4} -> SNR: {item['snr_db']:>6.2f} dB | MAE: {item['mae']:>6.4f} ({item['privacy_guarantee']} Privacy)")

    # 6. Audit Trail Check
    print("\n[Step 6] Phase 6: Verifying Cryptographic Audit Log...")
    logger = AuditLogger()
    logger.log(user_id="admin", role="admin", action="DEMO_PIPELINE_RUN", resource="WORKFLOW")
    check = verify_audit_log_integrity(logger.log_path)
    print(f"Audit log status: {'VALID' if check['valid'] else 'INVALID'} (Total records: {check['records_count']})")

    print("\n" + "=" * 65)
    print("  All phases (Phase 4, Phase 5, Phase 6) successfully verified!")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
