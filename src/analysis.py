"""
What drove lap time in one race: fuel, tyre wear and driver pace.

All three are estimated together, by least squares on clean laps:

    lap time = driver pace + compound offset
             + fuel effect x laps remaining
             + wear rate (per compound) x tyre age

They have to be estimated together. Fuel burns off and tyres wear at the
same time, so fitting lap time against lap number alone (or correcting for
fuel using tyre age) mixes the two up. Fitting the whole field at once
separates them, because drivers pit on different laps.
"""

import numpy as np

from feature_engineering import clean_laps

MIN_COMPOUND_LAPS = 40


def fit_race_effects(laps):
    """
    Fit one race's tidy lap table.

    Returns a dict with:
        fuel_effect_per_lap: seconds per lap of fuel still to burn
        degradation: {compound: {"rate_per_lap", "offset", "n_laps"}}
        not_fitted: {compound: n_laps} for compounds with too few clean laps
        driver_pace: {driver: seconds vs the field average; negative is faster}
        total_laps, residual_std
    """
    clean = clean_laps(laps)
    counts = clean["Compound"].value_counts()
    compounds = [c for c in counts.index if counts[c] >= MIN_COMPOUND_LAPS]
    if not compounds:
        raise ValueError("No compound has enough clean laps to fit.")
    reference = compounds[0]
    clean = clean[clean["Compound"].isin(compounds)]
    drivers = sorted(clean["Driver"].unique())

    age = clean["TyreLife"].to_numpy(dtype=float)
    compound = clean["Compound"].to_numpy()
    columns = [(clean["Driver"].to_numpy() == d).astype(float) for d in drivers]
    columns += [(compound == c).astype(float) for c in compounds if c != reference]
    columns.append((clean["TotalLaps"] - clean["LapNumber"]).to_numpy(dtype=float))
    columns += [np.where(compound == c, age, 0.0) for c in compounds]
    matrix = np.column_stack(columns)
    target = clean["LapTimeSec"].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    residuals = target - matrix @ coefficients

    position = iter(coefficients)
    pace = {driver: float(next(position)) for driver in drivers}
    offsets = {reference: 0.0}
    offsets.update({c: float(next(position)) for c in compounds if c != reference})
    fuel_effect = float(next(position))
    rates = {c: float(next(position)) for c in compounds}

    field_average = float(np.mean(list(pace.values())))
    return {
        "race": str(laps["Race"].iloc[0]),
        "total_laps": int(laps["TotalLaps"].iloc[0]),
        "fuel_effect_per_lap": fuel_effect,
        "degradation": {
            c: {"rate_per_lap": rates[c], "offset": offsets[c], "n_laps": int(counts[c])}
            for c in compounds
        },
        "not_fitted": {c: int(n) for c, n in counts.items() if c not in compounds},
        "driver_pace": dict(sorted(
            ((d, p - field_average) for d, p in pace.items()), key=lambda item: item[1])),
        "field_average_pace": field_average,
        "residual_std": float(np.sqrt(residuals @ residuals / (len(target) - rank))),
    }


def expected_stint_time(effects, compound, stint_length, start_lap, base_pace=None):
    """
    Expected lap times for a stint on new tyres starting on `start_lap`.

    Args:
        base_pace: lap time on the reference compound with no fuel and no
            wear; defaults to the field average for the race.
    """
    compound = compound.upper()
    if compound not in effects["degradation"]:
        available = ", ".join(sorted(effects["degradation"]))
        raise ValueError(f"No fit for {compound} in this race. Available: {available}.")
    if stint_length < 1 or start_lap < 1 or start_lap + stint_length - 1 > effects["total_laps"]:
        raise ValueError(
            f"Stint must be at least 1 lap and end by lap {effects['total_laps']}.")
    base_pace = effects["field_average_pace"] if base_pace is None else base_pace
    tyre = effects["degradation"][compound]
    lap_times = [
        base_pace + tyre["offset"] + tyre["rate_per_lap"] * age
        + effects["fuel_effect_per_lap"] * (effects["total_laps"] - (start_lap + age))
        for age in range(stint_length)
    ]
    return {"compound": compound, "stint_length": stint_length, "start_lap": start_lap,
            "lap_times": lap_times, "total_time": float(sum(lap_times))}
