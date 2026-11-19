import os
import json
import pytest
import numpy as np
from datetime import timedelta

from security.encryption import (
    generate_data_encryption_key,
    encrypt_csi_payload,
    decrypt_csi_payload,
    SecureCSIStorage
)
from security.auth import (
    hash_password,
    verify_password,
    create_access_token,
    verify_access_token,
    UserRole,
    Permission,
    check_user_permission
)
from security.privacy import add_differential_privacy_noise, evaluate_privacy_utility_tradeoff
from security.audit import AuditLogger, verify_audit_log_integrity


def test_encryption_roundtrip():
    key = generate_data_encryption_key()
    original_arr = np.random.randn(20, 10).astype(np.float32)

    encrypted = encrypt_csi_payload(original_arr, key)
    assert isinstance(encrypted, bytes)
    assert encrypted != original_arr.tobytes()

    decrypted_arr = decrypt_csi_payload(encrypted, key, as_numpy=True)
    assert np.allclose(original_arr, decrypted_arr)


def test_secure_storage(tmp_path):
    storage = SecureCSIStorage()
    target_path = str(tmp_path / "test_csi.npz")
    data = {"amp": np.random.randn(10, 5), "phase": np.random.randn(10, 5)}

    enc_path = storage.save_encrypted_npz(target_path, data)
    assert os.path.exists(enc_path)

    loaded = storage.load_encrypted_npz(enc_path)
    assert "amp" in loaded
    assert np.allclose(data["amp"], loaded["amp"])


def test_auth_and_passwords():
    pwd = "SecretPassword123!"
    hashed = hash_password(pwd)
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

    # JWT
    token = create_access_token({"sub": "admin", "role": "admin"})
    payload = verify_access_token(token)
    assert payload["sub"] == "admin"
    assert payload["role"] == "admin"

    # Expired token
    expired_token = create_access_token({"sub": "admin"}, expires_delta=timedelta(seconds=-10))
    with pytest.raises(ValueError):
        verify_access_token(expired_token)


def test_rbac_matrix():
    assert check_user_permission("admin", Permission.MANAGE_USERS) is True
    assert check_user_permission("operator", Permission.MANAGE_USERS) is False
    assert check_user_permission("operator", Permission.WRITE_CSI) is True
    assert check_user_permission("auditor", Permission.VIEW_AUDIT_LOGS) is True


def test_differential_privacy():
    raw_data = np.ones((50, 20))
    noisy = add_differential_privacy_noise(raw_data, epsilon=1.0, mechanism="laplace")
    assert noisy.shape == raw_data.shape
    assert not np.allclose(raw_data, noisy)

    # Low epsilon -> more noise, higher distortion
    noisy_strict = add_differential_privacy_noise(raw_data, epsilon=0.1)
    noisy_loose = add_differential_privacy_noise(raw_data, epsilon=10.0)
    assert np.mean(np.abs(noisy_strict - raw_data)) > np.mean(np.abs(noisy_loose - raw_data))


def test_audit_log_tamper_detection(tmp_path):
    log_file = str(tmp_path / "test_audit.jsonl")
    logger = AuditLogger(log_file)

    logger.log(user_id="user1", role="admin", action="CREATE", resource="RECORD_1")
    logger.log(user_id="user2", role="operator", action="UPDATE", resource="RECORD_2")

    # Clean verification
    status = verify_audit_log_integrity(log_file)
    assert status["valid"] is True
    assert status["records_count"] == 2

    # Intentionally tamper with record 1 in file
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    tampered_rec = json.loads(lines[0])
    tampered_rec["user_id"] = "attacker"  # Alter user_id without re-signing hash
    lines[0] = json.dumps(tampered_rec) + "\n"

    with open(log_file, "w") as f:
        f.writelines(lines)

    tampered_status = verify_audit_log_integrity(log_file)
    assert tampered_status["valid"] is False
    assert "tampering detected" in tampered_status["reason"].lower()
