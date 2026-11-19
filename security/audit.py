"""
WiFiSense - Tamper-Evident Audit Logging Module
Implements append-only cryptographic hash-chained audit trails (SHA-256) for security compliance.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AuditRecord:
    index: int
    timestamp: str
    user_id: str
    role: str
    action: str
    resource: str
    status: str
    details: Dict[str, Any]
    previous_hash: str
    hash: str = ""

    def calculate_hash(self) -> str:
        payload = f"{self.index}|{self.timestamp}|{self.user_id}|{self.role}|{self.action}|{self.resource}|{self.status}|{json.dumps(self.details, sort_keys=True)}|{self.previous_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    """
    Append-only cryptographically verifiable audit trail.
    """

    def __init__(self, log_path: str = "security/audit_trail.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.last_hash = "0" * 64
        self.count = 0
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_obj = json.loads(lines[-1])
                    self.last_hash = last_obj.get("hash", "0" * 64)
                    self.count = len(lines)

    def log(
        self,
        user_id: str,
        role: str,
        action: str,
        resource: str,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """
        Appends an event to the cryptographically chained audit log.
        """
        self.count += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        record = AuditRecord(
            index=self.count,
            timestamp=now_iso,
            user_id=user_id,
            role=role,
            action=action,
            resource=resource,
            status=status,
            details=details or {},
            previous_hash=self.last_hash
        )
        record.hash = record.calculate_hash()
        self.last_hash = record.hash

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        return record

    def get_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves recent audit records.
        """
        if not os.path.exists(self.log_path):
            return []
        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records[-limit:]


def verify_audit_log_integrity(log_path: str = "security/audit_trail.jsonl") -> Dict[str, Any]:
    """
    Verifies cryptographic hash chain integrity of the entire audit trail.

    Returns:
        Dict indicating validity, total records verified, and any tamper indices.
    """
    if not os.path.exists(log_path):
        return {"valid": True, "records_count": 0, "message": "Log file does not exist yet."}

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    expected_prev_hash = "0" * 64
    for idx, r in enumerate(records):
        rec = AuditRecord(
            index=r["index"],
            timestamp=r["timestamp"],
            user_id=r["user_id"],
            role=r["role"],
            action=r["action"],
            resource=r["resource"],
            status=r["status"],
            details=r.get("details", {}),
            previous_hash=r["previous_hash"],
            hash=r["hash"]
        )

        # Check chain link
        if rec.previous_hash != expected_prev_hash:
            return {
                "valid": False,
                "broken_index": idx,
                "reason": f"Hash chain broken at index {idx}. Expected previous {expected_prev_hash}, found {rec.previous_hash}"
            }

        # Check payload hash integrity
        calculated = rec.calculate_hash()
        if calculated != rec.hash:
            return {
                "valid": False,
                "broken_index": idx,
                "reason": f"Data tampering detected at record {idx}. Stored hash {rec.hash} != computed hash {calculated}"
            }

        expected_prev_hash = rec.hash

    return {
        "valid": True,
        "records_count": len(records),
        "latest_hash": expected_prev_hash,
        "message": "Cryptographic chain integrity verified 100%."
    }
