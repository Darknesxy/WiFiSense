"""
WiFiSense - Database Layer
SQLite relational storage with parameterized queries, schema management, and credential seeding.
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from security.auth import hash_password, UserRole

DB_PATH = os.getenv("WIFISENSE_DB_PATH", "wifisense.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the database schema and seeds default administrative and operator roles.
    """
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    # CSI Capture Sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS csi_sessions (
        session_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        device_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)

    # CSI Records with Encrypted Storage
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS csi_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_id INTEGER NOT NULL,
        encrypted_payload BLOB NOT NULL,
        prediction TEXT,
        confidence REAL,
        privacy_epsilon REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)

    conn.commit()

    # Seed initial test accounts if empty
    cursor.execute("SELECT COUNT(*) FROM users;")
    count = cursor.fetchone()[0]

    if count == 0:
        now_iso = datetime.now(timezone.utc).isoformat()
        default_users = [
            ("admin", "Admin@WiFiSense2026", UserRole.ADMIN.value),
            ("operator", "Operator@123", UserRole.OPERATOR.value),
            ("analyst", "Analyst@123", UserRole.ANALYST.value),
            ("auditor", "Auditor@123", UserRole.AUDITOR.value),
        ]
        for uname, pwd, role in default_users:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (uname, hash_password(pwd), role, now_iso)
            )
        conn.commit()
        print("[Database] Initialized schema and seeded default accounts.")

    conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(username: str, password_plain: str, role: str) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()
    pwd_hash = hash_password(password_plain)
    cursor.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (username, pwd_hash, role, now_iso)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": user_id, "username": username, "role": role, "created_at": now_iso}


def save_csi_record(
    user_id: int, 
    encrypted_payload: bytes, 
    prediction: str, 
    confidence: float, 
    epsilon: float,
    session_id: Optional[str] = None
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        """
        INSERT INTO csi_records 
        (session_id, user_id, encrypted_payload, prediction, confidence, privacy_epsilon, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, user_id, encrypted_payload, prediction, confidence, epsilon, now_iso)
    )
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id


def list_csi_records(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, session_id, user_id, prediction, confidence, privacy_epsilon, created_at 
        FROM csi_records 
        ORDER BY id DESC LIMIT ?
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
