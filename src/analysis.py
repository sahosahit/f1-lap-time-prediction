"""
Race performance analysis: driver pace decomposition, fuel burn, tyre degradation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_fuel_burn(laps_df):
    """
    Estimate fuel burn effect (sec/lap improvement as fuel decreases).

    Args:
        laps_df: DataFrame with LapTimeSec and LapNumber columns

    Returns:
        Dict with fuel_effect (sec/lap), r2, total_improvement
    """
    df = laps_df.copy()
    X = df[['LapNumber']].values
    y = df['LapTimeSec'].values

    model = LinearRegression()
    model.fit(X, y)

    fuel_effect = -model.coef_[0]
    total_laps = df['LapNumber'].max()

    return {
        'fuel_effect_per_lap': fuel_effect,
        'r2': model.score(X, y),
        'total_improvement': fuel_effect * total_laps,
    }


def estimate_tyre_degradation(laps_df, fuel_effect=0.060):
    """
    Estimate tyre degradation rate per compound after removing fuel effects.

    Args:
        laps_df: DataFrame with LapTimeSec, TyreLife, Compound
        fuel_effect: Fuel burn improvement per lap (to subtract)

    Returns:
        Dict with degradation rate per compound
    """
    df = laps_df.copy()
    df['FuelCorrectedTime'] = df['LapTimeSec'] + fuel_effect * df['TyreLife']

    results = {}
    for compound in df['Compound'].unique():
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) < 5:
            continue

        X = compound_data[['TyreLife']].values
        y = compound_data['FuelCorrectedTime'].values

        model = LinearRegression()
        model.fit(X, y)

        results[compound] = {
            'degradation_rate': model.coef_[0],
            'r2': model.score(X, y),
            'n_laps': len(compound_data),
        }

    return results


def decompose_driver_pace(laps_df, fuel_effect=0.060):
    """
    Extract driver pace by removing fuel and tyre effects.

    Args:
        laps_df: DataFrame with LapTimeSec, TyreLife, LapNumber, Driver
        fuel_effect: Fuel burn improvement per lap

    Returns:
        DataFrame with driver pace rankings (delta vs field average)
    """
    df = laps_df.copy()
    df['CorrectedTime'] = df['LapTimeSec'] + fuel_effect * (df['LapNumber'] - 1)

    field_mean = df['CorrectedTime'].mean()
    driver_pace = df.groupby('Driver')['CorrectedTime'].mean().reset_index()
    driver_pace.columns = ['Driver', 'MeanCorrectedTime']
    driver_pace['PaceDelta'] = driver_pace['MeanCorrectedTime'] - field_mean
    driver_pace = driver_pace.sort_values('PaceDelta')

    return driver_pace


def simulate_stint(compound, stint_length, base_pace=99.0, fuel_effect=0.060, degradation_rates=None):
    """
    Monte Carlo stint simulation.

    Args:
        compound: Tyre compound name
        stint_length: Number of laps
        base_pace: Base lap time
        fuel_effect: Fuel burn improvement per lap
        degradation_rates: Dict with compound degradation rates

    Returns:
        Dict with mean, std, and lap times
    """
    if degradation_rates is None:
        degradation_rates = {'SOFT': 0.066, 'MEDIUM': 0.029, 'HARD': 0.016}

    deg_rate = degradation_rates.get(compound, 0.03)
    n_simulations = 500
    all_totals = []

    for _ in range(n_simulations):
        total = 0
        for lap in range(stint_length):
            lap_time = base_pace
            lap_time += deg_rate * lap
            lap_time -= fuel_effect * lap
            lap_time += np.random.normal(0, 0.2)
            total += lap_time
        all_totals.append(total)

    return {
        'mean': np.mean(all_totals),
        'std': np.std(all_totals),
        'compound': compound,
        'stint_length': stint_length,
    }
