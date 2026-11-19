"""
WiFiSense - Authentication & Access Control (RBAC) Module
Provides password hashing, JWT token lifecycle management, and Role-Based Access Control.
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Dict, List, Set, Any
from jose import jwt, JWTError

# Cryptographic secrets
SECRET_KEY = os.getenv("WIFISENSE_JWT_SECRET", "wifisense-super-secret-production-key-2026-xyz-987")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    AUDITOR = "auditor"


class Permission(str, Enum):
    READ_CSI = "read:csi"
    WRITE_CSI = "write:csi"
    RUN_INFERENCE = "inference:run"
    MANAGE_USERS = "users:manage"
    VIEW_AUDIT_LOGS = "audit:view"
    EXPORT_DATA = "data:export"


# Role-to-Permissions Mapping Matrix
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.READ_CSI,
        Permission.WRITE_CSI,
        Permission.RUN_INFERENCE,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.EXPORT_DATA
    },
    UserRole.ANALYST: {
        Permission.READ_CSI,
        Permission.RUN_INFERENCE,
        Permission.EXPORT_DATA
    },
    UserRole.OPERATOR: {
        Permission.READ_CSI,
        Permission.WRITE_CSI,
        Permission.RUN_INFERENCE
    },
    UserRole.AUDITOR: {
        Permission.VIEW_AUDIT_LOGS,
        Permission.READ_CSI
    }
}


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """
    Hashes a password with SHA-256 and cryptographic salt (or PBKDF2).
    Format: salt$hex_hash
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${hashed.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain text password against a stored salt$hex_hash string.
    """
    try:
        salt, expected_hash = hashed_password.split("$")
        computed = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
        return secrets.compare_digest(computed, expected_hash)
    except Exception:
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Encodes and signs a JSON Web Token (JWT) with expiration.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates a JWT token. Raises ValueError on expiration or invalid signature.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise ValueError(f"Invalid or expired JWT token: {str(e)}")


def check_user_permission(role: str, permission: Permission) -> bool:
    """
    Verifies whether a given role holds the requested permission.
    """
    try:
        user_role = UserRole(role.lower())
        return permission in ROLE_PERMISSIONS.get(user_role, set())
    except ValueError:
        return False
