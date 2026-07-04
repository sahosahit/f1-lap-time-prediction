"""Tests for ML model training and evaluation."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from model import train_model, evaluate_model, cross_validate_model, get_default_params


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    X = np.column_stack([
        np.random.randint(0, 20, n),      # DriverEncoded
        np.random.randint(0, 3, n),        # CompoundEncoded
        np.random.uniform(1, 30, n),       # TyreLife
        np.random.uniform(1, 900, n),      # TyreLifeSquared
        np.random.uniform(10, 55, n),      # FuelLoadProxy
    ])
    y = 95 + 0.05 * X[:, 2] - 0.02 * X[:, 4] + np.random.normal(0, 0.5, n)
    return X, y


class TestTrainModel:
    def test_random_forest(self, sample_data):
        X, y = sample_data
        model = train_model(X, y, model_type='random_forest')
        assert hasattr(model, 'predict')
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_gradient_boosting(self, sample_data):
        X, y = sample_data
        model = train_model(X, y, model_type='gradient_boosting')
        assert hasattr(model, 'predict')

    def test_linear_regression(self, sample_data):
        X, y = sample_data
        model = train_model(X, y, model_type='linear_regression')
        assert hasattr(model, 'predict')

    def test_invalid_model_type(self, sample_data):
        X, y = sample_data
        with pytest.raises(ValueError):
            train_model(X, y, model_type='invalid')


class TestEvaluateModel:
    def test_returns_metrics(self, sample_data):
        X, y = sample_data
        model = train_model(X[:80], y[:80])
        metrics = evaluate_model(model, X[80:], y[80:])
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0


class TestCrossValidate:
    def test_returns_cv_scores(self, sample_data):
        X, y = sample_data
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        result = cross_validate_model(model, X, y, cv=3)
        assert 'cv_mae_mean' in result
        assert 'cv_mae_std' in result
        assert len(result['cv_scores']) == 3


class TestGetDefaultParams:
    def test_random_forest_params(self):
        params = get_default_params('random_forest')
        assert params['n_estimators'] == 500
        assert params['max_depth'] == 20

    def test_unknown_returns_empty(self):
        params = get_default_params('unknown')
        assert params == {}
