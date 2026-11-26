"""Tests for turning race lap tables into a modelling dataset."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import CATEGORICAL, NUMERIC, TARGET, build_dataset, clean_laps, race_features
from races import RACES
from synthetic import make_race


@pytest.fixture(scope="module")
def race():
    return make_race()


def test_pit_laps_lap_one_and_neutralised_laps_are_removed(race):
    race = race.copy()
    race.loc[race["LapNumber"] == 30, "Neutralised"] = True
    clean = clean_laps(race)
    assert not (clean["IsPitInLap"] | clean["IsPitOutLap"]).any()
    assert not clean["LapNumber"].isin([1, 30]).any()


def test_a_very_slow_lap_is_dropped(race):
    race = race.copy()
    slow = (race["Driver"] == "D00") & (race["LapNumber"] == 5)
    race.loc[slow, "LapTimeSec"] *= 1.2
    assert clean_laps(race)[lambda t: (t["Driver"] == "D00") & (t["LapNumber"] == 5)].empty


def test_target_is_lap_time_as_a_percentage_of_the_race_median(race):
    features = race_features(race)
    assert features[TARGET].median() == pytest.approx(100.0, abs=0.01)
    row = features.iloc[0]
    assert row[TARGET] == pytest.approx(row["LapTimeSec"] / row["RefLapTime"] * 100)


def test_fuel_fraction_runs_from_nearly_one_to_zero(race):
    features = race_features(race)
    assert features["FuelFraction"].max() == pytest.approx(48 / 50)
    assert features["FuelFraction"].min() == 0.0


def test_tyre_life_is_split_by_compound(race):
    features = race_features(race)
    hard = features[features["Compound"] == "HARD"].iloc[5]
    assert hard["TyreLife_HARD"] == hard["TyreLife"]
    assert hard["TyreLife_MEDIUM"] == 0 and hard["TyreLife_SOFT"] == 0


def test_sector_times_are_not_features():
    assert not any("Sector" in name for name in CATEGORICAL + NUMERIC)


def test_committed_races_build_into_one_dataset():
    dataset = build_dataset()
    assert set(dataset["Race"]) == set(RACES)
    assert not dataset[CATEGORICAL + NUMERIC + [TARGET]].isna().any().any()
