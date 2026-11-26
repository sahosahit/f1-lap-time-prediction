"""Tests for the FastAPI endpoints."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi.testclient import TestClient

from api.main import app

RACE = "2024_austin"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_model_and_races(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["races"] == 16


def test_races_and_drivers_are_listed(client):
    assert RACE in client.get("/races").json()["races"]
    assert "VER" in client.get("/drivers").json()["drivers"]


def test_predict_returns_a_plausible_lap_time(client):
    body = {"driver": "VER", "compound": "hard", "tyre_life": 10, "lap_number": 30,
            "total_laps": 56, "reference_lap_time": 99.0}
    data = client.post("/predict", json=body).json()
    assert 95.0 < data["predicted_lap_time"] < 103.0
    assert data["relative_to_reference_pct"] == pytest.approx(
        data["predicted_lap_time"] / 99.0 * 100, abs=0.01)


def test_predict_rejects_an_unknown_compound_and_a_bad_lap(client):
    body = {"driver": "VER", "compound": "ULTRASOFT", "tyre_life": 10, "lap_number": 30,
            "total_laps": 56, "reference_lap_time": 99.0}
    assert client.post("/predict", json=body).status_code == 422
    body.update(compound="HARD", lap_number=60)
    assert client.post("/predict", json=body).status_code == 422


def test_analysis_reports_fuel_tyres_and_pace_for_a_race(client):
    data = client.get(f"/analysis/{RACE}").json()
    assert data["race"] == RACE
    assert 0.0 < data["fuel_effect_per_lap"] < 0.15
    assert all(info["n_laps"] >= 40 for info in data["degradation"].values())
    assert len(data["driver_pace"]) >= 15


def test_analysis_of_an_unknown_race_is_404(client):
    assert client.get("/analysis/2024_nowhere").status_code == 404


def test_stint_prediction(client):
    compound = next(iter(client.get(f"/analysis/{RACE}").json()["degradation"]))
    body = {"race": RACE, "compound": compound, "stint_length": 15, "start_lap": 20}
    data = client.post("/predict/stint", json=body).json()
    assert len(data["lap_times"]) == 15
    assert data["mean_lap_time"] == pytest.approx(data["total_time"] / 15, abs=0.01)


def test_stint_on_an_unfitted_compound_or_past_the_flag_is_400(client):
    analysis = client.get(f"/analysis/{RACE}").json()
    body = {"race": RACE, "compound": next(iter(analysis["degradation"])),
            "stint_length": 50, "start_lap": 40}
    assert client.post("/predict/stint", json=body).status_code == 400
    if analysis["not_fitted"]:
        body = {"race": RACE, "compound": next(iter(analysis["not_fitted"])),
                "stint_length": 10, "start_lap": 5}
        assert client.post("/predict/stint", json=body).status_code == 400
