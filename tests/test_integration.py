import pytest
import httpx

GATEWAY_URL = "http://localhost:8080"
AUTH = ("admin", "1qaz2wsx")

def test_healthz():
    response = httpx.get(f"{GATEWAY_URL}/healthz", auth=AUTH, headers={"Host": "gateway.local"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_positive():
    response = httpx.post(
        f"{GATEWAY_URL}/predict",
        json={"text": "this movie was absolutely fantastic"},
        auth=AUTH,
        headers={"Host": "gateway.local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["positive", "negative"]
    assert 0.0 < data["confidence"] <= 1.0

def test_predict_negative():
    response = httpx.post(
        f"{GATEWAY_URL}/predict",
        json={"text": "this movie was terrible and boring"},
        auth=AUTH,
        headers={"Host": "gateway.local"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["positive", "negative"]
    assert 0.0 < data["confidence"] <= 1.0

def test_predict_missing_text():
    response = httpx.post(
        f"{GATEWAY_URL}/predict",
        json={},
        auth=AUTH,
        headers={"Host": "gateway.local"}
    )
    assert response.status_code == 422

def test_unauthorized():
    response = httpx.post(
        f"{GATEWAY_URL}/predict",
        json={"text": "great movie"},
        headers={"Host": "gateway.local"}
    )
    assert response.status_code == 401

def test_predict_confidence_is_probability():
    response = httpx.post(
        f"{GATEWAY_URL}/predict",
        json={"text": "decent film"},
        auth=AUTH,
        headers={"Host": "gateway.local"}
    )
    data = response.json()
    assert data["confidence"] > 0.0
    assert data["confidence"] <= 1.0