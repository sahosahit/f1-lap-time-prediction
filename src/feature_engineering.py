"""
Turn race lap tables into a dataset for lap-time models.

Lap times at different circuits are on different scales (about 80 s at
Monza, 100 s at Austin), so the target is the lap time as a percentage of
that race's median clean lap. The models learn how lap time moves with
tyres, fuel and driver, not how long a lap of a given circuit is.

Sector times are deliberately not features: they are parts of the lap time
being predicted.
"""

import pandas as pd

from openf1_loader import load_race
from races import RACES

OUTLIER_FACTOR = 1.07
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
CATEGORICAL = ["Driver", "Compound"]
NUMERIC = ["TyreLife", "FuelFraction"] + [f"TyreLife_{c}" for c in COMPOUNDS]
TARGET = "RelativeLapTime"


def clean_laps(laps):
    """
    Laps that show the car's real pace: no pit in-laps or out-laps, no
    standing-start lap, no safety-car laps, and nothing slower than 107% of
    the driver's median (spins, damage, heavy traffic).
    """
    keep = ~(laps["IsPitInLap"] | laps["IsPitOutLap"] | laps["Neutralised"]
             | (laps["LapNumber"] == 1))
    kept = laps[keep]
    median = kept.groupby("Driver")["LapTimeSec"].transform("median")
    return kept[kept["LapTimeSec"] <= OUTLIER_FACTOR * median]


def add_model_inputs(frame):
    """Columns derived from Compound and TyreLife that every model needs."""
    frame = frame.copy()
    for compound in COMPOUNDS:
        frame[f"TyreLife_{compound}"] = frame["TyreLife"].where(frame["Compound"] == compound, 0.0)
    return frame


def race_features(laps):
    """Modelling rows for one race."""
    clean = clean_laps(laps).copy()
    reference = clean["LapTimeSec"].median()
    clean["RefLapTime"] = reference
    clean["FuelFraction"] = (clean["TotalLaps"] - clean["LapNumber"]) / clean["TotalLaps"]
    clean[TARGET] = clean["LapTimeSec"] / reference * 100.0
    clean["TyreLife"] = clean["TyreLife"].astype(float)
    columns = ["Race", "Driver", "LapNumber", "TotalLaps", "Compound", "TyreLife", "FuelFraction",
               "LapTimeSec", "RefLapTime", TARGET]
    return add_model_inputs(clean[columns]).reset_index(drop=True)


def build_dataset(slugs=None):
    """All committed races as one dataset."""
    slugs = list(RACES) if slugs is None else slugs
    return pd.concat([race_features(load_race(slug)) for slug in slugs], ignore_index=True)
