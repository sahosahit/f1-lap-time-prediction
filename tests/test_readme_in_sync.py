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


def test_block_names_the_best_model_and_both_baselines():
    evaluation = {
        "n_races": 2, "n_laps": 200, "best_model": "linear",
        "overall": {name: {"mae_pct": 0.5 + i / 10, "mae_sec": 0.4 + i / 10, "n_laps": 200,
                           "n_races": 2}
                    for i, name in enumerate(["linear", "random_forest", "gradient_boosting",
                                              "race_median", "driver_average"])},
        "per_race": {name: {"2024_a": {"mae_sec": 0.5, "mae_pct": 0.5, "n_laps": 100},
                            "2024_b": {"mae_sec": 0.7, "mae_pct": 0.7, "n_laps": 100}}
                     for name in ["linear", "random_forest", "gradient_boosting",
                                  "race_median", "driver_average"]},
    }
    block = render_results_block(evaluation)
    assert "| Linear (ridge) **(best)** | 0.40 s | 0.50% |" in block
    assert "Every lap equals the race median" in block
    assert "| A 2024 | 100 | 0.50 | 0.50 | 0.50 |" in block
