"""Tests for FastAPI endpoints."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["data_loaded"] is True


class TestPredictEndpoint:
    def test_predict_without_model_returns_503(self, client):
        response = client.post("/predict", json={
            "driver_encoded": 17,
            "compound": "MEDIUM",
            "tyre_life": 10.0,
            "fuel_load_proxy": 40.0,
        })
        if response.status_code == 503:
            assert "Model not loaded" in response.json()["detail"]
        else:
            assert response.status_code == 200
            assert "predicted_lap_time" in response.json()


class TestAnalysisEndpoints:
    def test_pace_analysis(self, client):
        response = client.get("/analysis/pace")
        assert response.status_code == 200
        data = response.json()
        assert "rankings" in data
        assert len(data["rankings"]) > 0

    def test_degradation_analysis(self, client):
        response = client.get("/analysis/degradation")
        assert response.status_code == 200
        data = response.json()
        assert "compounds" in data

    def test_fuel_analysis(self, client):
        response = client.get("/analysis/fuel")
        assert response.status_code == 200
        data = response.json()
        assert "fuel_effect_per_lap" in data


class TestStintPrediction:
    def test_predict_stint(self, client):
        response = client.post("/predict/stint", json={
            "compound": "HARD",
            "stint_length": 20,
            "base_pace": 99.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["compound"] == "HARD"
        assert data["stint_length"] == 20
        assert data["mean_total_time"] > 0


class TestDriversEndpoint:
    def test_list_drivers(self, client):
        response = client.get("/drivers")
        assert response.status_code == 200
        data = response.json()
        assert len(data["drivers"]) > 0
        assert "abbreviation" in data["drivers"][0]
        assert "encoded" in data["drivers"][0]
