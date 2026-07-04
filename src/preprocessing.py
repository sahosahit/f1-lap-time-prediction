"""
Lap data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np


def clean_laps(laps_df):
    """
    Clean raw lap data: filter accurate laps, remove pit laps, convert times.

    Args:
        laps_df: Raw DataFrame from FastF1

    Returns:
        Cleaned DataFrame with LapTimeSec column
    """
    df = laps_df.copy()

    df = df[df['IsAccurate'] == True]
    df = df[~df['PitInTime'].notna()]
    df = df[~df['PitOutTime'].notna()]

    df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
    df['Sector1Sec'] = df['Sector1Time'].dt.total_seconds()
    df['Sector2Sec'] = df['Sector2Time'].dt.total_seconds()
    df['Sector3Sec'] = df['Sector3Time'].dt.total_seconds()

    return df


def filter_outliers(df, column='LapTimeSec', std_threshold=3):
    """
    Remove statistical outliers beyond N standard deviations.

    Args:
        df: DataFrame
        column: Column to check for outliers
        std_threshold: Number of standard deviations for cutoff

    Returns:
        Filtered DataFrame
    """
    mean = df[column].mean()
    std = df[column].std()
    lower = mean - std_threshold * std
    upper = mean + std_threshold * std
    return df[(df[column] >= lower) & (df[column] <= upper)]
