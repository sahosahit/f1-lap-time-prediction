# F1 Lap Time Prediction & Race Performance Modeling

Analyzing the 2024 United States Grand Prix at Circuit of the Americas (COTA) using Formula 1 telemetry data from the FastF1 API.

The goal is to build a race performance modeling pipeline that estimates key factors influencing lap time, including driver race pace, tyre degradation, fuel burn effects, and strategy implications.

Combines machine learning, statistical modeling, and motorsport performance analysis to simulate race conditions and extract insights similar to those used by F1 performance and strategy engineers.

## Key Results

| Factor | Estimated Effect |
|--------|------------------|
| Fuel Burn | 0.060 sec/lap improvement |
| Soft Tyre Degradation | 0.066 sec/lap |
| Medium Tyre Degradation | 0.029 sec/lap |
| Hard Tyre Degradation | 0.016 sec/lap |
| Lap Time Prediction (Hold-out MAE) | 0.38 sec |
| Lap Time Prediction (5-fold CV MAE) | ~1.0 sec |
| Best Model | Random Forest (tuned: max_depth=20, min_samples_leaf=5, n_estimators=500) |

**Note:** Cross-validation reveals the true generalization error (~1.0 sec) is higher than the single hold-out split (0.38 sec), indicating the default model overfits. The tuned model with regularization generalizes better.

## Project Workflow

```
Race Data Extraction (FastF1 API)
        ↓
Feature Engineering (tyre, fuel, driver encoding)
        ↓
Lap Time Prediction (ML Models + Cross-Validation + Hyperparameter Tuning)
        ↓
Driver Pace Decomposition
        ↓
Fuel Burn Modeling
        ↓
Tyre Degradation Estimation
        ↓
Strategy Insights (Monte Carlo Stint Simulation)
```

## Project Structure

```
f1-lap-time-prediction/
├── notebooks/
│   ├── data_extraction.ipynb              # FastF1 data pull & cleaning
│   ├── feature_engineering.ipynb          # Feature creation
│   ├── model_training_and_evaluation.ipynb # ML models, CV, hyperparameter tuning
│   ├── driver_pace_decomposition.ipynb    # Driver performance analysis
│   ├── fuel_and_tyre_model.ipynb          # Fuel burn & tyre degradation
│   ├── stint_performance_stimulator.ipynb # Strategy simulation
│   └── final_report.ipynb                 # Summary & visualizations
├── data/
│   ├── raw/cota_2024_laps.csv
│   └── processed/cota_2024_features.csv
├── cache/                                  # FastF1 API cache
├── src/
├── requirements.txt
└── README.md
```

## ML Model Training & Evaluation

### Features Used
- `DriverEncoded` — Categorical driver identifier
- `CompoundEncoded` — Tyre compound (SOFT=0, MEDIUM=1, HARD=2)
- `TyreLife` — Laps on current tyre set
- `TyreLifeSquared` — Non-linear tyre degradation (captures cliff effect)
- `FuelLoadProxy` — Approximated remaining fuel (56 - LapNumber)

### Model Comparison

| Model | Hold-out MAE | Hold-out RMSE | 5-Fold CV MAE |
|-------|-------------|---------------|----------------|
| Linear Regression | 0.700 sec | 0.919 sec | 0.832 ± 0.140 |
| Random Forest (default) | 0.379 sec | 0.562 sec | 1.008 ± 0.133 |
| Gradient Boosting | 0.391 sec | 0.559 sec | 0.945 ± 0.169 |
| **Random Forest (tuned)** | **0.433 sec** | - | **0.984** |

### Hyperparameter Tuning (GridSearchCV)
```
Best Parameters: max_depth=20, min_samples_leaf=5, n_estimators=500
```

The tuned model trades slightly worse hold-out performance for better generalization (less overfitting).

### Key Insight: Overfitting Detection

The default Random Forest achieves 0.38 MAE on the hold-out set but ~1.0 MAE on cross-validation. This gap indicates overfitting to the specific train/test split. Cross-validation provides a more honest estimate of real-world performance.

## Driver Race Pace

Driver performance estimated by removing fuel effects and tyre degradation from lap times:

| Driver | Pace Delta (vs field) |
|--------|----------------------|
| LEC | -1.24 sec (fastest) |
| SAI | -1.07 sec |
| NOR | -0.98 sec |
| VER | -0.88 sec |
| BOT | +1.28 sec (slowest) |

## Tyre Degradation

Degradation rates estimated via linear regression on fuel-corrected lap times:

- **Soft:** 0.066 sec/lap (fastest degradation, limited data: 3 laps)
- **Medium:** 0.029 sec/lap (balanced)
- **Hard:** 0.016 sec/lap (most stable)

## Fuel Burn Effect

- Fuel improvement: **0.060 sec/lap** (as fuel burns, car gets lighter and faster)
- Race-long improvement: ~3.3 seconds over 56 laps

## Strategy Insights

20-lap stint simulation results:
- Soft tyres: 1989.14 sec total (fastest initial pace, highest degradation)
- Medium tyres: 1982.11 sec total (best compromise)
- Hard tyres: 1979.64 sec total (best overall for long stints)

**Conclusion:** Hard tyres optimal for longer stints; soft tyres suited only for short aggressive stints.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/
```

**Note:** FastF1 API caches data locally. First run will download ~50MB of telemetry data.

## Technologies

- **Python** — Primary language
- **FastF1** — Official F1 telemetry API
- **scikit-learn** — ML models (Random Forest, Gradient Boosting, Linear Regression), GridSearchCV, cross-validation
- **pandas / NumPy** — Data processing
- **matplotlib / seaborn** — Visualization
- **SciPy** — Statistical analysis

## Future Work

- Non-linear tyre degradation modeling (piecewise regression for cliff effect)
- Traffic and DRS effects on lap time
- Track temperature influence
- Multi-race datasets for stronger generalization
- Temporal cross-validation (train on early laps, test on late laps)

## Data Source

Race data from the [FastF1 API](https://theoehrly.github.io/Fast-F1/) — official Formula 1 telemetry and timing data.
