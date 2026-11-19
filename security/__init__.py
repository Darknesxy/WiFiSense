"""
WiFiSense Security & Privacy Package
Implements AES-256 / Fernet data encryption, JWT auth & RBAC, Differential Privacy, and tamper-evident audit logs.
"""

from .encryption import (
    generate_data_encryption_key,
    encrypt_csi_payload,
    decrypt_csi_payload,
    SecureCSIStorage
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    verify_access_token,
    UserRole,
    Permission,
    check_user_permission
)
from .privacy import (
    add_differential_privacy_noise,
    evaluate_privacy_utility_tradeoff,
    anonymize_csi_metadata
)
from .audit import (
    AuditLogger,
    AuditRecord,
    verify_audit_log_integrity
)

__all__ = [
    "generate_data_encryption_key",
    "encrypt_csi_payload",
    "decrypt_csi_payload",
    "SecureCSIStorage",
    "hash_password",
    "verify_password",
    "create_access_token",
    "verify_access_token",
    "UserRole",
    "Permission",
    "check_user_permission",
    "add_differential_privacy_noise",
    "evaluate_privacy_utility_tradeoff",
    "anonymize_csi_metadata",
    "AuditLogger",
    "AuditRecord",
    "verify_audit_log_integrity"
]
