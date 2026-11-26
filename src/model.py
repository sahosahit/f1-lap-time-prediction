"""
Lap-time models and how they are evaluated.

Evaluation is leave-one-race-out: each race is predicted by a model fitted
on all the other races. Splitting laps at random would put laps from the
same stint on both sides of the split, and a model can then score well by
memorising the stint. Holding out a whole race asks the real question:
does what was learned carry over to a race the model has not seen?

Two baselines give the scores meaning:
- race_median:    every lap equals the race's median lap (predicts 100%);
- driver_average: every lap equals that driver's usual pace relative to
                  the field, learned from the other races.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import CATEGORICAL, NUMERIC, TARGET, add_model_inputs

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lap_time_model.joblib")
MODEL_KINDS = ("linear", "random_forest", "gradient_boosting")
BASELINES = ("race_median", "driver_average")


def make_model(kind):
    """
    A scikit-learn pipeline: drivers and compounds one-hot encoded (a driver
    is a category, not a number on a scale), numeric inputs passed through.
    A driver the model has never seen is treated as an average driver.
    """
    estimators = {
        "linear": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    }
    if kind not in estimators:
        raise ValueError(f"Unknown model kind: {kind}. Choose from {MODEL_KINDS}.")
    encode = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numbers", "passthrough", NUMERIC),
    ])
    return Pipeline([("encode", encode), ("estimator", estimators[kind]())])


def _score(test, predicted_pct):
    error_pct = test[TARGET].to_numpy() - predicted_pct
    return {
        "mae_pct": float(np.abs(error_pct).mean()),
        "mae_sec": float(np.abs(error_pct * test["RefLapTime"].to_numpy() / 100.0).mean()),
        "mean_error_pct": float(error_pct.mean()),
        "n_laps": int(len(test)),
    }


def leave_one_race_out(dataset, kind):
    """Score each race with a model fitted on every other race."""
    scores = {}
    for race in sorted(dataset["Race"].unique()):
        train, test = dataset[dataset["Race"] != race], dataset[dataset["Race"] == race]
        model = make_model(kind).fit(train, train[TARGET])
        scores[race] = _score(test, model.predict(test))
    return scores


def baseline_leave_one_race_out(dataset, baseline):
    """Score each race with a baseline that uses only the other races."""
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline}. Choose from {BASELINES}.")
    scores = {}
    for race in sorted(dataset["Race"].unique()):
        train, test = dataset[dataset["Race"] != race], dataset[dataset["Race"] == race]
        if baseline == "race_median":
            predicted = np.full(len(test), 100.0)
        else:
            usual_pace = train.groupby("Driver")[TARGET].mean()
            predicted = test["Driver"].map(usual_pace).fillna(100.0).to_numpy()
        scores[race] = _score(test, predicted)
    return scores


def summarise(scores):
    """Pool per-race scores, weighting each race by its number of laps."""
    laps = sum(s["n_laps"] for s in scores.values())
    return {
        "mae_pct": sum(s["mae_pct"] * s["n_laps"] for s in scores.values()) / laps,
        "mae_sec": sum(s["mae_sec"] * s["n_laps"] for s in scores.values()) / laps,
        "n_laps": laps,
        "n_races": len(scores),
    }


def train_final(dataset, kind):
    """Fit one model on every race, for serving predictions."""
    return make_model(kind).fit(dataset, dataset[TARGET])


def predict_seconds(model, driver, compound, tyre_life, lap_number, total_laps,
                    reference_lap_time):
    """
    Predicted lap time in seconds.

    Args:
        reference_lap_time: the race's typical clean lap in seconds. The model
            predicts relative to it, so it must come from the caller.
    """
    row = add_model_inputs(pd.DataFrame([{
        "Driver": driver,
        "Compound": compound.upper(),
        "TyreLife": float(tyre_life),
        "FuelFraction": (total_laps - lap_number) / total_laps,
    }]))
    return float(model.predict(row)[0] * reference_lap_time / 100.0)


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path=MODEL_PATH):
    """The saved model, or None if it has not been trained yet."""
    return joblib.load(path) if os.path.exists(path) else None
