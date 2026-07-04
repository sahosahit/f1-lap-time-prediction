"""Tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from preprocessing import filter_outliers


class TestFilterOutliers:
    def test_removes_outliers(self):
        data = pd.DataFrame({'LapTimeSec': [100, 101, 100, 99, 100, 200, 50]})
        filtered = filter_outliers(data, column='LapTimeSec', std_threshold=2)
        assert len(filtered) < len(data)
        assert 200 not in filtered['LapTimeSec'].values

    def test_keeps_normal_data(self):
        data = pd.DataFrame({'LapTimeSec': [100, 101, 99, 100, 101, 100]})
        filtered = filter_outliers(data, column='LapTimeSec', std_threshold=3)
        assert len(filtered) == len(data)

    def test_custom_column(self):
        data = pd.DataFrame({'speed': [200, 201, 199, 500]})
        filtered = filter_outliers(data, column='speed', std_threshold=2)
        assert 500 not in filtered['speed'].values
