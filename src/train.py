"""
Training script for F1 Lap Time Prediction model.
Trains on processed COTA 2024 data and saves model to models/ directory.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import train_model, evaluate_model, cross_validate_model, save_model, get_default_params

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("F1 LAP TIME PREDICTION - MODEL TRAINING")
    print("=" * 60)

    data_path = os.path.join(DATA_DIR, 'cota_2024_features.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    feature_cols = ['DriverEncoded', 'CompoundEncoded', 'TyreLife', 'FuelLoadProxy']
    df['TyreLifeSquared'] = df['TyreLife'] ** 2
    feature_cols_with_sq = feature_cols + ['TyreLifeSquared']

    X = df[feature_cols_with_sq].values
    y = df['LapTimeSec'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print(f"\n{'='*60}")
    print("TRAINING MODELS")
    print(f"{'='*60}")

    results = {}
    for model_type in ['random_forest', 'gradient_boosting', 'linear_regression']:
        print(f"\n  {model_type}:")
        params = get_default_params(model_type)
        model = train_model(X_train, y_train, model_type=model_type, params=params)

        metrics = evaluate_model(model, X_test, y_test)
        print(f"    MAE:  {metrics['mae']:.3f} sec")
        print(f"    RMSE: {metrics['rmse']:.3f} sec")
        print(f"    R2:   {metrics['r2']:.3f}")

        from sklearn.base import clone
        cv_model = clone(model) if model_type != 'linear_regression' else model
        cv = cross_validate_model(cv_model, X, y, cv=5)
        print(f"    5-fold CV MAE: {cv['cv_mae_mean']:.3f} +/- {cv['cv_mae_std']:.3f}")

        results[model_type] = {**metrics, **cv}

    best_type = min(results, key=lambda k: results[k]['mae'])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_type} (MAE={results[best_type]['mae']:.3f} sec)")
    print(f"{'='*60}")

    best_model = train_model(X_train, y_train, model_type=best_type)
    model_path = save_model(best_model)
    print(f"Model saved to: {model_path}")

    results_path = os.path.join(RESULTS_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
