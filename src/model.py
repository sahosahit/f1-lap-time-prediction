"""
ML model training, evaluation, and prediction for lap time estimation.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def get_default_params(model_type):
    """Return tuned hyperparameters for each model type."""
    params = {
        'random_forest': {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_leaf': 5,
            'random_state': 42,
        },
        'gradient_boosting': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        },
        'linear_regression': {},
    }
    return params.get(model_type, {})


def train_model(X_train, y_train, model_type='random_forest', params=None):
    """
    Train a regression model.

    Args:
        X_train: Training features
        y_train: Training targets
        model_type: 'random_forest', 'gradient_boosting', or 'linear_regression'
        params: Optional hyperparameters (uses defaults if None)

    Returns:
        Trained model
    """
    if params is None:
        params = get_default_params(model_type)

    models = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'linear_regression': LinearRegression,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(models.keys())}")

    model = models[model_type](**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.

    Returns:
        Dict with MAE, RMSE, R2 metrics
    """
    predictions = model.predict(X_test)
    return {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions),
    }


def cross_validate_model(model, X, y, cv=5):
    """
    Run cross-validation and return scores.

    Returns:
        Dict with mean and std of MAE across folds
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    return {
        'cv_mae_mean': -scores.mean(),
        'cv_mae_std': scores.std(),
        'cv_scores': (-scores).tolist(),
    }


def save_model(model, filename='lap_time_model.joblib'):
    """Save trained model to disk."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(filename='lap_time_model.joblib'):
    """Load trained model from disk."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)
