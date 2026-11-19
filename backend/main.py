"""
WiFiSense - FastAPI Application Backend
REST endpoints for authentication, secure CSI ingestion, real-time PyTorch inference, and security audits.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import (
    init_db,
    get_user_by_username,
    create_user,
    save_csi_record,
    list_csi_records
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
from security.encryption import (
    generate_data_encryption_key,
    encrypt_csi_payload,
    decrypt_csi_payload,
    SecureCSIStorage
)
from security.privacy import (
    add_differential_privacy_noise,
    evaluate_privacy_utility_tradeoff
)
from security.audit import AuditLogger, verify_audit_log_integrity
from models.cnn_models import CSI_1D_CNN
from preprocessing.cleaning import preprocess_csi_packet_stream
from datasets.generate_synthetic_csi import ACTIVITIES

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_model()
    audit_logger.log(
        user_id="SYSTEM",
        role="SYSTEM",
        action="SERVER_STARTUP",
        resource="FASTAPI_BACKEND",
        status="SUCCESS",
        details={"status": "Online", "version": "1.0.0"}
    )
    yield

app = FastAPI(
    title="WiFiSense API",
    description="Secure Deep Learning & Human Activity Recognition Platform for WiFi Channel State Information",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services & audit logger
audit_logger = AuditLogger("security/audit_trail.jsonl")
secure_storage = SecureCSIStorage()

# In-memory cached PyTorch model
MODEL_PATH = "models/cnn_1d_model.pth"
_loaded_model: Optional[CSI_1D_CNN] = None


def get_model() -> CSI_1D_CNN:
    global _loaded_model
    if _loaded_model is None:
        model = CSI_1D_CNN(in_channels=56, num_classes=len(ACTIVITIES))
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location="cpu")
                model.load_state_dict(state)
                print(f"[Model] Loaded pre-trained checkpoint from {MODEL_PATH}")
            except Exception as e:
                print(f"[Model] Notice: Using initialized weights ({e})")
        model.eval()
        _loaded_model = model
    return _loaded_model





# --- Security & Auth Dependencies ---

def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    token = authorization.split(" ")[1]
    try:
        payload = verify_access_token(token)
        return payload
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


def require_role(allowed_roles: List[UserRole]):
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role")
        if not user_role or user_role not in [r.value for r in allowed_roles]:
            audit_logger.log(
                user_id=current_user.get("sub", "unknown"),
                role=user_role or "unknown",
                action="RBAC_ACCESS_DENIED",
                resource="ENDPOINT",
                status="FAILED",
                details={"required_roles": [r.value for r in allowed_roles], "user_role": user_role}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user
    return role_checker


# --- Schemas ---

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    expires_in_minutes: int = 60


class CSIPredictionRequest(BaseModel):
    csi_amplitude: List[List[float]] = Field(..., description="CSI Amplitude matrix [time_steps, subcarriers]")
    apply_dp: bool = False
    dp_epsilon: float = 1.0


class CSIPredictionResponse(BaseModel):
    predicted_activity: str
    confidence: float
    probabilities: Dict[str, float]
    inference_latency_ms: float
    dp_applied: bool


class SecureUploadRequest(BaseModel):
    csi_amplitude: List[List[float]]
    session_id: Optional[str] = None
    privacy_epsilon: float = 1.0


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "WiFiSense API",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_classes": ACTIVITIES
    }


@app.post("/auth/login", response_model=LoginResponse)
def login(creds: LoginRequest):
    user = get_user_by_username(creds.username)
    if not user or not verify_password(creds.password, user["password_hash"]):
        audit_logger.log(
            user_id=creds.username,
            role="GUEST",
            action="LOGIN_ATTEMPT",
            resource="/auth/login",
            status="FAILED",
            details={"reason": "Invalid credentials"}
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")

    token = create_access_token({"sub": user["username"], "user_id": user["id"], "role": user["role"]})
    
    audit_logger.log(
        user_id=user["username"],
        role=user["role"],
        action="LOGIN_SUCCESS",
        resource="/auth/login",
        status="SUCCESS",
        details={"user_id": user["id"]}
    )

    return LoginResponse(
        access_token=token,
        username=user["username"],
        role=user["role"]
    )


@app.get("/auth/me")
def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    return user


@app.post("/predict/csi", response_model=CSIPredictionResponse)
def predict_activity(
    req: CSIPredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    import time
    t0 = time.perf_counter()

    raw_arr = np.array(req.csi_amplitude, dtype=np.float32)
    if raw_arr.ndim != 2:
        raise HTTPException(status_code=400, detail="csi_amplitude must be a 2D matrix (time_steps, subcarriers)")

    # 1. Privacy Protection (if enabled)
    if req.apply_dp:
        raw_arr = add_differential_privacy_noise(raw_arr, epsilon=req.dp_epsilon)

    # 2. Cleaning & Normalization
    cleaned = preprocess_csi_packet_stream(raw_arr, normalize=True)

    # 3. Model Inference (PyTorch 1D-CNN)
    # Target shape for 1D-CNN: (1, n_subcarriers, time_steps)
    model = get_model()
    tensor_in = torch.tensor(cleaned.T, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    top_idx = int(np.argmax(probs))
    pred_label = ACTIVITIES[top_idx]
    confidence = float(probs[top_idx])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    prob_dict = {act: float(probs[i]) for i, act in enumerate(ACTIVITIES)}

    # Audit logging
    audit_logger.log(
        user_id=current_user.get("sub", "anon"),
        role=current_user.get("role", "none"),
        action="INFERENCE_EXECUTION",
        resource="/predict/csi",
        status="SUCCESS",
        details={"prediction": pred_label, "confidence": round(confidence, 4), "dp_applied": req.apply_dp}
    )

    return CSIPredictionResponse(
        predicted_activity=pred_label,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        inference_latency_ms=round(latency_ms, 2),
        dp_applied=req.apply_dp
    )


@app.post("/data/upload-secure")
def upload_secure_csi(
    req: SecureUploadRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR]))
):
    raw_arr = np.array(req.csi_amplitude, dtype=np.float32)
    
    # Run rapid inference
    model = get_model()
    cleaned = preprocess_csi_packet_stream(raw_arr, normalize=True)
    tensor_in = torch.tensor(cleaned.T, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor_in), dim=1).squeeze(0).numpy()
    top_idx = int(np.argmax(probs))
    pred_label = ACTIVITIES[top_idx]
    conf = float(probs[top_idx])

    # Encrypt payload using AES-256 / Fernet key
    encrypted_blob = encrypt_csi_payload(raw_arr, secure_storage.key)

    record_id = save_csi_record(
        user_id=current_user.get("user_id", 1),
        encrypted_payload=encrypted_blob,
        prediction=pred_label,
        confidence=round(conf, 4),
        epsilon=req.privacy_epsilon,
        session_id=req.session_id
    )

    audit_logger.log(
        user_id=current_user.get("sub", "anon"),
        role=current_user.get("role", "none"),
        action="DATA_ENCRYPTED_INGESTION",
        resource="csi_records",
        status="SUCCESS",
        details={"record_id": record_id, "shape": list(raw_arr.shape), "prediction": pred_label}
    )

    return {
        "status": "encrypted_and_stored",
        "record_id": record_id,
        "prediction": pred_label,
        "confidence": conf
    }


@app.get("/data/records")
def get_stored_records(
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.OPERATOR, UserRole.ANALYST]))
):
    records = list_csi_records(limit=25)
    return {"records": records, "count": len(records)}


@app.get("/security/audit")
def get_audit_trail(
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.AUDITOR]))
):
    records = audit_logger.get_records(limit=limit)
    return {"audit_trail": records, "count": len(records)}


@app.post("/security/audit/verify")
def verify_audit_trail(
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN, UserRole.AUDITOR]))
):
    verification_report = verify_audit_log_integrity(audit_logger.log_path)
    return verification_report


@app.post("/security/privacy/evaluate")
def evaluate_dp(
    req: CSIPredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    sample = np.array(req.csi_amplitude, dtype=np.float32)
    tradeoff = evaluate_privacy_utility_tradeoff(sample)
    return {"tradeoff_analysis": tradeoff}
