"""
Feature engineering for lap time prediction.
"""

import pandas as pd
import numpy as np


COMPOUND_MAP = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}


def encode_compounds(series):
    """Map tyre compound names to numeric values."""
    return series.map(COMPOUND_MAP).fillna(1).astype(int)


def encode_drivers(series):
    """Label encode driver abbreviations."""
    unique_drivers = sorted(series.unique())
    mapping = {driver: i for i, driver in enumerate(unique_drivers)}
    return series.map(mapping)


def create_features(laps_df, total_laps=56):
    """
    Create ML features from cleaned lap data.

    Args:
        laps_df: Cleaned DataFrame with lap timing data
        total_laps: Total race laps (for fuel proxy calculation)

    Returns:
        DataFrame with features and target (LapTimeSec)
    """
    df = laps_df.copy()

    if 'Abbreviation' in df.columns:
        df['Driver'] = df['Abbreviation']

    df['DriverEncoded'] = encode_drivers(df['Driver'])
    df['CompoundEncoded'] = encode_compounds(df['Compound'])
    df['TyreLife'] = df['TyreLife'].astype(float) if 'TyreLife' in df.columns else df.groupby(['Driver', 'Stint']).cumcount().astype(float)
    df['TyreLifeSquared'] = df['TyreLife'] ** 2
    df['FuelLoadProxy'] = total_laps - df['LapNumber'].astype(float)

    feature_cols = ['DriverEncoded', 'CompoundEncoded', 'TyreLife', 'TyreLifeSquared', 'FuelLoadProxy']
    target_col = 'LapTimeSec'

    result = df[feature_cols + [target_col, 'Driver', 'Compound']].copy()
    result = result.dropna(subset=feature_cols + [target_col])

    return result
