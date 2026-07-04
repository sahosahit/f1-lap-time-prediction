"""
FastAPI service for F1 Lap Time Prediction.
Serves predictions and race analysis via REST API.
"""

import sys
import os
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from model import load_model, evaluate_model
from feature_engineering import COMPOUND_MAP
from analysis import (
    estimate_fuel_burn, estimate_tyre_degradation,
    decompose_driver_pace, simulate_stint
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state['model'] = load_model()
    data_path = os.path.join(DATA_DIR, 'cota_2024_features.csv')
    if os.path.exists(data_path):
        _state['data'] = pd.read_csv(data_path)
    else:
        _state['data'] = None
    yield
    _state.clear()


app = FastAPI(
    title="F1 Lap Time Prediction API",
    description="ML-based lap time prediction and race performance analysis (COTA 2024)",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    driver_encoded: int
    compound: str = "MEDIUM"
    tyre_life: float = 10.0
    fuel_load_proxy: float = 40.0


class StintRequest(BaseModel):
    compound: str = "MEDIUM"
    stint_length: int = 20
    base_pace: float = 99.0


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    data_rows: int


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_state.get('model') is not None,
        data_loaded=_state.get('data') is not None,
        data_rows=len(_state['data']) if _state.get('data') is not None else 0,
    )


@app.post("/predict")
async def predict_lap_time(request: PredictRequest):
    model = _state.get('model')
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first: python src/train.py")

    compound_encoded = COMPOUND_MAP.get(request.compound.upper(), 1)
    tyre_life_squared = request.tyre_life ** 2

    features = np.array([[
        request.driver_encoded,
        compound_encoded,
        request.tyre_life,
        tyre_life_squared,
        request.fuel_load_proxy,
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_lap_time": round(float(prediction), 3),
        "inputs": {
            "driver_encoded": request.driver_encoded,
            "compound": request.compound,
            "tyre_life": request.tyre_life,
            "fuel_load_proxy": request.fuel_load_proxy,
        },
    }


@app.get("/analysis/pace")
async def driver_pace_analysis():
    data = _state.get('data')
    if data is None:
        raise HTTPException(status_code=503, detail="No race data loaded")

    df = data.copy()
    if 'LapNumber' not in df.columns:
        df['LapNumber'] = df.groupby('Driver').cumcount() + 1

    pace = decompose_driver_pace(df)
    return {
        "race": "COTA 2024",
        "rankings": [
            {
                "driver": row['Driver'],
                "pace_delta": round(row['PaceDelta'], 3),
            }
            for _, row in pace.iterrows()
        ],
    }


@app.get("/analysis/degradation")
async def degradation_analysis():
    data = _state.get('data')
    if data is None:
        raise HTTPException(status_code=503, detail="No race data loaded")

    results = estimate_tyre_degradation(data)
    return {
        "race": "COTA 2024",
        "compounds": {
            compound: {
                "degradation_rate_per_lap": round(info['degradation_rate'], 4),
                "r2": round(info['r2'], 3),
                "sample_laps": info['n_laps'],
            }
            for compound, info in results.items()
        },
    }


@app.get("/analysis/fuel")
async def fuel_analysis():
    data = _state.get('data')
    if data is None:
        raise HTTPException(status_code=503, detail="No race data loaded")

    if 'LapNumber' not in data.columns:
        data['LapNumber'] = data.groupby('Driver').cumcount() + 1

    result = estimate_fuel_burn(data)
    return {
        "race": "COTA 2024",
        "fuel_effect_per_lap": round(result['fuel_effect_per_lap'], 4),
        "r2": round(result['r2'], 3),
        "total_improvement_over_race": round(result['total_improvement'], 2),
    }


@app.post("/predict/stint")
async def predict_stint(request: StintRequest):
    result = simulate_stint(
        compound=request.compound.upper(),
        stint_length=request.stint_length,
        base_pace=request.base_pace,
    )
    return {
        "compound": result['compound'],
        "stint_length": result['stint_length'],
        "mean_total_time": round(result['mean'], 2),
        "std": round(result['std'], 2),
        "mean_lap_time": round(result['mean'] / result['stint_length'], 3),
    }


@app.get("/drivers")
async def list_drivers():
    data = _state.get('data')
    if data is None:
        raise HTTPException(status_code=503, detail="No race data loaded")

    drivers = data[['Driver', 'DriverEncoded']].drop_duplicates().sort_values('DriverEncoded')
    return {
        "drivers": [
            {"abbreviation": row['Driver'], "encoded": int(row['DriverEncoded'])}
            for _, row in drivers.iterrows()
        ],
    }
