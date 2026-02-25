import pytest
import torch
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

os.environ["MODEL_DIR"] = "models/model"
os.environ["KSERVE_URL"] = "http://mock-kserve/predict"
os.environ["JAEGER_ENDPOINT"] = "localhost:4317"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/app"))


@pytest.fixture
def client():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoConfig.from_pretrained") as mock_config:

        # Mock tokenizer
        mock_tok_instance = MagicMock()
        mock_tok_instance.return_value = {"input_ids": torch.tensor([[101, 2054, 1037, 2307, 3185, 102]])}
        mock_tokenizer.return_value = mock_tok_instance

        # Mock config
        mock_cfg_instance = MagicMock()
        mock_cfg_instance.id2label = {0: "negative", 1: "positive"}
        mock_config.return_value = mock_cfg_instance

        from gateway import app
        with TestClient(app) as c:
            yield c


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_positive(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"logits": [[-2.5, 3.1]]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        response = client.post("/predict", json={"text": "this movie was absolutely fantastic"})
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert data["label"] == "positive"
        assert 0.0 < data["confidence"] <= 1.0


def test_predict_negative(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"logits": [[3.1, -2.5]]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        response = client.post("/predict", json={"text": "this movie was terrible and boring"})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "negative"
        assert 0.0 < data["confidence"] <= 1.0


def test_predict_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_empty_text(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"logits": [[0.1, 0.9]]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 200


def test_predict_kserve_error(client):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        response = client.post("/predict", json={"text": "some review"})
        assert response.status_code == 500


def test_predict_confidence_is_probability(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"logits": [[-1.0, 1.0]]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        response = client.post("/predict", json={"text": "decent film"})
        data = response.json()
        assert data["confidence"] > 0.5