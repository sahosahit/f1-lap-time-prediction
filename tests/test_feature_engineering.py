"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from feature_engineering import encode_compounds, encode_drivers, create_features, COMPOUND_MAP


class TestEncodeCompounds:
    def test_correct_mapping(self):
        series = pd.Series(['SOFT', 'MEDIUM', 'HARD', 'MEDIUM'])
        result = encode_compounds(series)
        assert result.tolist() == [0, 1, 2, 1]

    def test_unknown_compound_defaults_to_1(self):
        series = pd.Series(['UNKNOWN'])
        result = encode_compounds(series)
        assert result.tolist() == [1]


class TestEncodeDrivers:
    def test_label_encoding(self):
        series = pd.Series(['VER', 'HAM', 'NOR', 'VER'])
        result = encode_drivers(series)
        assert result[0] == result[3]  # VER same encoding
        assert len(set(result)) == 3   # 3 unique drivers

    def test_sorted_encoding(self):
        series = pd.Series(['ZHO', 'ALB', 'VER'])
        result = encode_drivers(series)
        assert result.iloc[1] < result.iloc[2]  # ALB < VER alphabetically


class TestCreateFeatures:
    def test_output_columns(self):
        df = pd.DataFrame({
            'Driver': ['VER'] * 5,
            'Compound': ['MEDIUM'] * 5,
            'TyreLife': [1.0, 2.0, 3.0, 4.0, 5.0],
            'LapNumber': [1, 2, 3, 4, 5],
            'LapTimeSec': [100.0, 99.5, 99.8, 100.1, 100.3],
            'Stint': [1] * 5,
        })
        result = create_features(df, total_laps=56)
        assert 'DriverEncoded' in result.columns
        assert 'CompoundEncoded' in result.columns
        assert 'TyreLifeSquared' in result.columns
        assert 'FuelLoadProxy' in result.columns

    def test_fuel_proxy_decreases(self):
        df = pd.DataFrame({
            'Driver': ['VER'] * 5,
            'Compound': ['MEDIUM'] * 5,
            'TyreLife': [1.0, 2.0, 3.0, 4.0, 5.0],
            'LapNumber': [1, 2, 3, 4, 5],
            'LapTimeSec': [100.0, 99.5, 99.8, 100.1, 100.3],
            'Stint': [1] * 5,
        })
        result = create_features(df, total_laps=56)
        fuel = result['FuelLoadProxy'].values
        assert all(fuel[i] > fuel[i+1] for i in range(len(fuel)-1))

    def test_tyre_life_squared(self):
        df = pd.DataFrame({
            'Driver': ['VER'] * 3,
            'Compound': ['HARD'] * 3,
            'TyreLife': [2.0, 4.0, 6.0],
            'LapNumber': [10, 11, 12],
            'LapTimeSec': [100.0, 100.1, 100.2],
            'Stint': [1] * 3,
        })
        result = create_features(df)
        assert result['TyreLifeSquared'].iloc[0] == 4.0
        assert result['TyreLifeSquared'].iloc[1] == 16.0
