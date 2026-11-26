"""
FastAPI service for F1 lap-time prediction and race analysis.

On start-up it loads the saved model, or fits one from the committed race
data if none has been saved, so the service works straight after a clone.
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from analysis import expected_stint_time, fit_race_effects
from feature_engineering import build_dataset
from model import load_model, predict_seconds, train_final
from openf1_loader import load_race
from races import RACES

EVALUATION_PATH = os.path.join(BASE_DIR, "results", "evaluation.json")
_state = {}


def _best_model_kind():
    if os.path.exists(EVALUATION_PATH):
        with open(EVALUATION_PATH) as handle:
            return json.load(handle)["best_model"]
    return "linear"


@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset = build_dataset()
    _state["drivers"] = sorted(dataset["Driver"].unique())
    _state["model"] = load_model() or train_final(dataset, _best_model_kind())
    _state["effects"] = {}
    yield
    _state.clear()


app = FastAPI(
    title="F1 Lap Time Prediction API",
    description="Lap-time prediction and per-race fuel, tyre and pace analysis (2024 season)",
    version="2.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    driver: str = Field(description="Three-letter code, e.g. VER. Unknown drivers are "
                                    "treated as an average driver.")
    compound: Literal["SOFT", "MEDIUM", "HARD"]
    tyre_life: float = Field(ge=0, le=80, description="Laps already done on this set")
    lap_number: int = Field(ge=1)
    total_laps: int = Field(ge=1, le=100)
    reference_lap_time: float = Field(gt=0, description="Typical clean race lap at this "
                                                         "circuit, in seconds")

    @model_validator(mode="before")
    @classmethod
    def _upper_case_compound(cls, data):
        if isinstance(data, dict) and isinstance(data.get("compound"), str):
            data = {**data, "compound": data["compound"].upper()}
        return data

    @model_validator(mode="after")
    def _lap_within_race(self):
        if self.lap_number > self.total_laps:
            raise ValueError("lap_number cannot be greater than total_laps")
        return self


class StintRequest(BaseModel):
    race: str
    compound: str
    stint_length: int
    start_lap: int
    base_pace: Optional[float] = None


def _effects(race):
    if race not in RACES:
        raise HTTPException(status_code=404, detail=f"Unknown race. Choose from {sorted(RACES)}.")
    if race not in _state["effects"]:
        _state["effects"][race] = fit_race_effects(load_race(race))
    return _state["effects"][race]


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": _state.get("model") is not None,
            "races": len(RACES)}


@app.get("/races")
async def list_races():
    return {"races": sorted(RACES)}


@app.get("/drivers")
async def list_drivers():
    return {"drivers": _state["drivers"]}


@app.post("/predict")
async def predict_lap_time(request: PredictRequest):
    seconds = predict_seconds(
        _state["model"], request.driver.upper(), request.compound, request.tyre_life,
        request.lap_number, request.total_laps, request.reference_lap_time,
    )
    return {
        "predicted_lap_time": round(seconds, 3),
        "relative_to_reference_pct": round(seconds / request.reference_lap_time * 100, 3),
        "inputs": request.model_dump(),
    }


@app.get("/analysis/{race}")
async def race_analysis(race: str):
    effects = _effects(race)
    return {
        "race": race,
        "fuel_effect_per_lap": round(effects["fuel_effect_per_lap"], 4),
        "degradation": {
            compound: {"rate_per_lap": round(info["rate_per_lap"], 4),
                       "offset": round(info["offset"], 3), "n_laps": info["n_laps"]}
            for compound, info in effects["degradation"].items()
        },
        "not_fitted": effects["not_fitted"],
        "driver_pace": {d: round(p, 3) for d, p in effects["driver_pace"].items()},
        "residual_std": round(effects["residual_std"], 3),
    }


@app.post("/predict/stint")
async def predict_stint(request: StintRequest):
    effects = _effects(request.race)
    try:
        stint = expected_stint_time(effects, request.compound, request.stint_length,
                                    request.start_lap, request.base_pace)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {
        "race": request.race,
        "compound": stint["compound"],
        "stint_length": stint["stint_length"],
        "start_lap": stint["start_lap"],
        "total_time": round(stint["total_time"], 2),
        "mean_lap_time": round(stint["total_time"] / stint["stint_length"], 3),
        "lap_times": [round(t, 3) for t in stint["lap_times"]],
    }
