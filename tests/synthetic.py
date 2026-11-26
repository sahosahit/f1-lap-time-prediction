"""Synthetic races with known fuel, tyre and driver effects."""

import numpy as np
import pandas as pd

COLUMNS = ["Race", "Driver", "LapNumber", "LapTimeSec", "Stint", "Compound", "TyreLife",
           "FittedNew", "IsPitInLap", "IsPitOutLap", "Neutralised", "TotalLaps"]
FUEL = 0.05
RATES = {"HARD": 0.04, "MEDIUM": 0.08, "SOFT": 0.12}
OFFSETS = {"HARD": 0.0, "MEDIUM": -0.4, "SOFT": -0.8}


def make_race(race="synthetic", base=90.0, total_laps=50, n_drivers=16, seed=0, noise=0.2,
              soft_drivers=()):
    """Staggered one-stop races: MEDIUM then HARD (or SOFT then HARD for `soft_drivers`)."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_drivers):
        first = "SOFT" if index in soft_drivers else "MEDIUM"
        pit_lap = 15 + index
        for lap in range(1, total_laps + 1):
            stint = 1 if lap <= pit_lap else 2
            compound = first if stint == 1 else "HARD"
            age = lap - 1 if stint == 1 else lap - pit_lap - 1
            lap_time = (base + 0.1 * index + OFFSETS[compound] + FUEL * (total_laps - lap)
                        + RATES[compound] * age + rng.normal(0, noise))
            rows.append([race, f"D{index:02d}", lap, lap_time, stint, compound, age, True,
                         lap == pit_lap, lap == pit_lap + 1, False, total_laps])
    return pd.DataFrame(rows, columns=COLUMNS)
