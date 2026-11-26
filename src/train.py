"""
Evaluate every model and baseline with leave-one-race-out, write
results/evaluation.json, then fit the best model on all races and save it.

    python src/train.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import build_dataset
from model import (
    BASELINES, MODEL_KINDS, baseline_leave_one_race_out, leave_one_race_out, save_model,
    summarise, train_final,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVALUATION_PATH = os.path.join(RESULTS_DIR, "evaluation.json")
README_PATH = os.path.join(BASE_DIR, "README.md")
BLOCK_START, BLOCK_END = "<!-- results:start -->", "<!-- results:end -->"
NAMES = {
    "linear": "Linear (ridge)",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "driver_average": "Baseline: each driver at their usual pace",
    "race_median": "Baseline: Every lap equals the race median",
}


def race_name(slug):
    year, _, place = slug.partition("_")
    return f"{place.replace('_', ' ').title()} {year}"


def render_results_block(evaluation):
    """The README results section, built only from results/evaluation.json."""
    best = evaluation["best_model"]
    lines = [
        f"Leave-one-race-out over {evaluation['n_races']} races and "
        f"{evaluation['n_laps']:,} clean laps: each race is predicted by a model fitted on "
        "the other races.",
        "",
        "| Method | Mean absolute error per lap | As a share of lap time |",
        "|---|---|---|",
    ]
    for name in ("gradient_boosting", "linear", "random_forest", "driver_average", "race_median"):
        score = evaluation["overall"][name]
        label = NAMES[name] + (" **(best)**" if name == best else "")
        lines.append(f"| {label} | {score['mae_sec']:.2f} s | {score['mae_pct']:.2f}% |")
    lines += [
        "",
        f"Per race, mean absolute error in seconds per lap ({NAMES[best].lower()} against "
        "the two baselines):",
        "",
        "| Race | Laps | Model | Driver's usual pace | Race median |",
        "|---|---|---|---|---|",
    ]
    for slug, score in evaluation["per_race"][best].items():
        usual = evaluation["per_race"]["driver_average"][slug]["mae_sec"]
        median = evaluation["per_race"]["race_median"][slug]["mae_sec"]
        lines.append(f"| {race_name(slug)} | {score['n_laps']} | {score['mae_sec']:.2f} "
                     f"| {usual:.2f} | {median:.2f} |")
    return "\n".join(lines)


def read_results_block(readme_text):
    start = readme_text.index(BLOCK_START) + len(BLOCK_START)
    return readme_text[start:readme_text.index(BLOCK_END)].strip("\n")


def write_results_block(block):
    with open(README_PATH) as handle:
        text = handle.read()
    start = text.index(BLOCK_START) + len(BLOCK_START)
    end = text.index(BLOCK_END)
    with open(README_PATH, "w") as handle:
        handle.write(f"{text[:start]}\n{block}\n{text[end:]}")


def evaluate(dataset):
    """Leave-one-race-out scores for every model and baseline."""
    per_race = {kind: leave_one_race_out(dataset, kind) for kind in MODEL_KINDS}
    per_race.update({b: baseline_leave_one_race_out(dataset, b) for b in BASELINES})
    overall = {name: summarise(scores) for name, scores in per_race.items()}
    best = min(MODEL_KINDS, key=lambda kind: overall[kind]["mae_pct"])
    return {
        "n_races": int(dataset["Race"].nunique()),
        "n_laps": int(len(dataset)),
        "best_model": best,
        "overall": overall,
        "per_race": per_race,
    }


def main():
    dataset = build_dataset()
    evaluation = evaluate(dataset)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(EVALUATION_PATH, "w") as handle:
        json.dump(evaluation, handle, indent=2)
        handle.write("\n")

    write_results_block(render_results_block(evaluation))

    print(f"{evaluation['n_laps']} laps from {evaluation['n_races']} races, leave-one-race-out:")
    for name, score in evaluation["overall"].items():
        print(f"  {name:18} {score['mae_pct']:.3f}%  {score['mae_sec']:.3f} s per lap")
    path = save_model(train_final(dataset, evaluation["best_model"]))
    print(f"Best model: {evaluation['best_model']} -> {path}")


if __name__ == "__main__":
    main()
