"""Tests for the per-race fuel, tyre and driver-pace analysis."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import expected_stint_time, fit_race_effects
from synthetic import FUEL, RATES, make_race


@pytest.fixture(scope="module")
def effects():
    return fit_race_effects(make_race(soft_drivers=(0, 1, 2, 3)))


def test_fuel_effect_is_separated_from_tyre_wear(effects):
    assert effects["fuel_effect_per_lap"] == pytest.approx(FUEL, abs=0.01)


def test_wear_rate_per_compound_is_recovered(effects):
    for compound in ("HARD", "MEDIUM", "SOFT"):
        assert effects["degradation"][compound]["rate_per_lap"] == pytest.approx(
            RATES[compound], abs=0.02)
        assert effects["degradation"][compound]["n_laps"] >= 40


def test_compound_with_too_few_laps_is_reported_not_fitted():
    effects = fit_race_effects(make_race(soft_drivers=(0,)))   # one short soft stint
    assert "SOFT" not in effects["degradation"]
    assert 0 < effects["not_fitted"]["SOFT"] < 40


def test_driver_pace_is_relative_to_the_field_and_ordered(effects):
    pace = effects["driver_pace"]
    assert sum(pace.values()) == pytest.approx(0.0, abs=1e-6)
    assert pace["D00"] < pace["D08"] < pace["D15"]
    assert pace["D15"] - pace["D00"] == pytest.approx(1.5, abs=0.3)


def test_expected_stint_time_adds_wear_and_fuel(effects):
    short = expected_stint_time(effects, "HARD", stint_length=10, start_lap=30, base_pace=90.0)
    long = expected_stint_time(effects, "HARD", stint_length=20, start_lap=30, base_pace=90.0)
    assert long["total_time"] > 2 * short["total_time"] - 5
    assert long["lap_times"][-1] > long["lap_times"][0] - 1.0
    assert len(long["lap_times"]) == 20


def test_stint_on_an_unfitted_compound_is_refused():
    effects = fit_race_effects(make_race(soft_drivers=(0,)))
    with pytest.raises(ValueError, match="SOFT"):
        expected_stint_time(effects, "SOFT", 10, 1, 90.0)


def test_stint_must_fit_inside_the_race(effects):
    with pytest.raises(ValueError):
        expected_stint_time(effects, "HARD", stint_length=30, start_lap=40, base_pace=90.0)
