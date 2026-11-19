import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db
from security.auth import create_access_token


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_classes" in data


def test_login_flow(client):
    # Valid credentials
    res = client.post("/auth/login", json={"username": "admin", "password": "Admin@WiFiSense2026"})
    assert res.status_code == 200
    token_data = res.json()
    assert "access_token" in token_data
    assert token_data["role"] == "admin"

    # Invalid credentials
    res_bad = client.post("/auth/login", json={"username": "admin", "password": "WrongPassword"})
    assert res_bad.status_code == 401


def test_authenticated_inference(client):
    token = create_access_token({"sub": "admin", "role": "admin", "user_id": 1})
    headers = {"Authorization": f"Bearer {token}"}

    sample_csi = np.random.randn(100, 56).tolist()
    res = client.post(
        "/predict/csi",
        headers=headers,
        json={"csi_amplitude": sample_csi, "apply_dp": False}
    )
    assert res.status_code == 200
    pred_data = res.json()
    assert "predicted_activity" in pred_data
    assert "confidence" in pred_data
    assert "probabilities" in pred_data


def test_unauthorized_rejection(client):
    sample_csi = np.random.randn(100, 56).tolist()
    res = client.post("/predict/csi", json={"csi_amplitude": sample_csi})
    assert res.status_code == 401
