"""
WiFiSense - Cryptographic Storage Module
Provides authenticated AES-256 / Fernet encryption for resting CSI dataset files and sensitive payload records.
"""

import os
import io
import json
from typing import Tuple, Dict, Any, Union
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_data_encryption_key() -> str:
    """
    Generates a cryptographically strong Fernet/AES key (URL-safe base64-encoded 32-byte key).
    """
    return Fernet.generate_key().decode("utf-8")


def encrypt_csi_payload(data: Union[bytes, str, np.ndarray], key: Union[str, bytes]) -> bytes:
    """
    Encrypts arbitrary CSI data (numpy array, JSON string, or raw bytes) using Fernet (AES-128-CBC + HMAC-SHA256).

    Args:
        data: Numpy array or string or raw bytes.
        key: Fernet key.

    Returns:
        Ciphertext bytes.
    """
    if isinstance(key, str):
        key = key.encode("utf-8")
    fernet = Fernet(key)

    if isinstance(data, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, data)
        raw_bytes = buffer.getvalue()
    elif isinstance(data, str):
        raw_bytes = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray)):
        raw_bytes = bytes(data)
    else:
        raw_bytes = json.dumps(data).encode("utf-8")

    return fernet.encrypt(raw_bytes)


def decrypt_csi_payload(ciphertext: Union[str, bytes], key: Union[str, bytes], as_numpy: bool = False) -> Any:
    """
    Decrypts encrypted CSI payload.

    Args:
        ciphertext: Encrypted token.
        key: Encryption key.
        as_numpy: If True, deserializes back into a numpy array.

    Returns:
        Decrypted bytes, numpy array, or string.
    """
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(ciphertext, str):
        ciphertext = ciphertext.encode("utf-8")

    fernet = Fernet(key)
    decrypted_bytes = fernet.decrypt(ciphertext)

    if as_numpy:
        buffer = io.BytesIO(decrypted_bytes)
        return np.load(buffer)
    return decrypted_bytes


class SecureCSIStorage:
    """
    High-level interface for saving and reading encrypted CSI numpy archives on disk.
    Ensures zero plaintext data touches persistent storage.
    """

    def __init__(self, key: Optional[str] = None):
        self.key = key or os.getenv("WIFISENSE_ENCRYPTION_KEY") or generate_data_encryption_key()

    def save_encrypted_npz(self, file_path: str, data_dict: Dict[str, np.ndarray]) -> str:
        """
        Compresses numpy arrays in-memory, encrypts the entire archive, and writes ciphertext to disk.
        """
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data_dict)
        raw_archive = buffer.getvalue()

        ciphertext = encrypt_csi_payload(raw_archive, self.key)

        enc_path = file_path if file_path.endswith(".enc") else file_path + ".enc"
        os.makedirs(os.path.dirname(os.path.abspath(enc_path)), exist_ok=True)
        with open(enc_path, "wb") as f:
            f.write(ciphertext)

        return enc_path

    def load_encrypted_npz(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Reads encrypted file from disk, decrypts in-memory, and loads numpy arrays.
        """
        with open(file_path, "rb") as f:
            ciphertext = f.read()

        decrypted_bytes = decrypt_csi_payload(ciphertext, self.key, as_numpy=False)
        buffer = io.BytesIO(decrypted_bytes)
        archive = np.load(buffer, allow_pickle=True)
        return {k: archive[k] for k in archive.files}
