"""Tests for the lap-time models and their leave-one-race-out evaluation."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import race_features
from model import (
    BASELINES, MODEL_KINDS, baseline_leave_one_race_out, leave_one_race_out, load_model,
    make_model, predict_seconds, save_model, summarise, train_final,
)
from synthetic import make_race


@pytest.fixture(scope="module")
def dataset():
    races = [make_race(f"race{i}", base=80.0 + 5 * i, seed=i) for i in range(4)]
    return pd.concat([race_features(r) for r in races], ignore_index=True)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_every_model_kind_fits_and_predicts(dataset, kind):
    model = make_model(kind).fit(dataset, dataset["RelativeLapTime"])
    assert len(model.predict(dataset.head(5))) == 5


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        make_model("magic")


def test_each_race_is_scored_by_a_model_that_never_saw_it(dataset):
    scores = leave_one_race_out(dataset, "linear")
    assert set(scores) == {"race0", "race1", "race2", "race3"}
    assert all(s["n_laps"] > 0 and s["mae_sec"] > 0 for s in scores.values())


def test_held_out_race_does_not_influence_its_own_score(dataset):
    tampered = dataset.copy()
    held_out = tampered["Race"] == "race3"
    tampered.loc[held_out, "RelativeLapTime"] += 5.0
    before = leave_one_race_out(dataset, "linear")
    after = leave_one_race_out(tampered, "linear")
    # Same predictions for race3, targets 5 points higher.
    assert after["race3"]["mean_error_pct"] - before["race3"]["mean_error_pct"] == pytest.approx(5.0)


def test_linear_model_beats_both_baselines_when_the_world_is_linear(dataset):
    model = summarise(leave_one_race_out(dataset, "linear"))
    for baseline in BASELINES:
        assert model["mae_pct"] < summarise(baseline_leave_one_race_out(dataset, baseline))["mae_pct"]


def test_summary_weights_races_by_laps():
    scores = {"a": {"mae_pct": 1.0, "mae_sec": 1.0, "n_laps": 100},
              "b": {"mae_pct": 3.0, "mae_sec": 2.0, "n_laps": 300}}
    assert summarise(scores) == {"mae_pct": 2.5, "mae_sec": 1.75, "n_laps": 400, "n_races": 2}


def test_unseen_driver_can_still_be_predicted(dataset):
    model = train_final(dataset, "linear")
    seconds = predict_seconds(model, driver="NEW", compound="HARD", tyre_life=10,
                              lap_number=20, total_laps=50, reference_lap_time=90.0)
    assert 85.0 < seconds < 95.0


def test_older_tyres_and_more_fuel_are_slower(dataset):
    model = train_final(dataset, "linear")
    def lap(**changes):
        inputs = dict(driver="D00", compound="HARD", tyre_life=10, lap_number=30,
                      total_laps=50, reference_lap_time=90.0)
        return predict_seconds(model, **{**inputs, **changes})
    assert lap(tyre_life=25) > lap(tyre_life=10)
    assert lap(lap_number=5) > lap(lap_number=45)


def test_model_round_trips_through_disk(dataset, tmp_path):
    model = train_final(dataset, "linear")
    path = save_model(model, tmp_path / "m.joblib")
    assert load_model(path).predict(dataset.head(3)) == pytest.approx(model.predict(dataset.head(3)))
    assert load_model(tmp_path / "missing.joblib") is None


def test_a_saved_model_from_an_older_version_is_ignored(tmp_path):
    import joblib
    from sklearn.linear_model import LinearRegression
    import numpy as np
    old = LinearRegression().fit(np.zeros((3, 5)), np.zeros(3))   # expects 5 bare numbers
    path = tmp_path / "old.joblib"
    joblib.dump(old, path)
    assert load_model(path) is None


def test_driver_plus_fuel_baseline_beats_driver_alone(dataset):
    plain = summarise(baseline_leave_one_race_out(dataset, "driver_average"))
    with_fuel = summarise(baseline_leave_one_race_out(dataset, "driver_plus_fuel"))
    assert with_fuel["mae_pct"] < plain["mae_pct"]
