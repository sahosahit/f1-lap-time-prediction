"""The README results must be exactly what results/evaluation.json says."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from train import EVALUATION_PATH, README_PATH, read_results_block, render_results_block


def test_readme_block_matches_the_evaluation_file():
    with open(EVALUATION_PATH) as handle:
        evaluation = json.load(handle)
    with open(README_PATH) as handle:
        assert read_results_block(handle.read()) == render_results_block(evaluation)


def test_block_labels_the_served_model_and_lists_every_baseline():
    names = ["linear", "random_forest", "gradient_boosting", "driver_plus_fuel",
             "driver_average", "race_median"]
    evaluation = {
        "n_races": 2, "n_laps": 200, "best_model": "gradient_boosting", "served_model": "linear",
        "overall": {name: {"mae_pct": 0.5 + i / 10, "mae_sec": 0.4 + i / 10, "n_laps": 200,
                           "n_races": 2} for i, name in enumerate(names)},
        "per_race": {name: {"2024_a": {"mae_sec": 0.5, "mae_pct": 0.5, "n_laps": 100},
                            "2024_b": {"mae_sec": 0.7, "mae_pct": 0.7, "n_laps": 100}}
                     for name in names},
    }
    block = render_results_block(evaluation)
    assert "| Linear (ridge) **(served by the API)** | 0.40 s | 0.50% |" in block
    assert "Baseline: driver's usual pace plus a fuel term" in block
    assert "Baseline: every lap equals the race median" in block
    assert "| A 2024 | 100 | 0.50 | 0.50 | 0.50 | 0.50 |" in block


def test_simplest_model_is_served_when_it_is_as_good_as_the_best():
    from train import choose_served_model
    close = {"linear": {"mae_sec": 0.571}, "gradient_boosting": {"mae_sec": 0.567},
             "random_forest": {"mae_sec": 0.593}}
    assert choose_served_model(close) == "linear"
    far = {"linear": {"mae_sec": 0.70}, "gradient_boosting": {"mae_sec": 0.567},
           "random_forest": {"mae_sec": 0.593}}
    assert choose_served_model(far) == "gradient_boosting"
