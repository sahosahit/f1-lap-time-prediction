"""Tests for turning OpenF1 records into the tidy lap table."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from openf1_loader import TIDY_COLUMNS, build_lap_table, is_dry, load_race
from races import RACES

DRIVERS = [
    {"driver_number": 1, "name_acronym": "VER"},
    {"driver_number": 44, "name_acronym": "HAM"},
]


def _laps(driver_number, n_laps, pit_out_laps=(), missing=()):
    return [
        {
            "driver_number": driver_number,
            "lap_number": lap,
            "lap_duration": None if lap in missing else 90.0 + 0.01 * lap,
            "is_pit_out_lap": lap in pit_out_laps,
        }
        for lap in range(1, n_laps + 1)
    ]


def _stint(driver_number, number, start, end, compound, age=0):
    return {
        "driver_number": driver_number,
        "stint_number": number,
        "lap_start": start,
        "lap_end": end,
        "compound": compound,
        "tyre_age_at_start": age,
    }


@pytest.fixture
def table():
    laps = _laps(1, 12, pit_out_laps={7}) + _laps(44, 12, pit_out_laps={6}, missing={10})
    stints = [
        _stint(1, 1, 1, 6, "SOFT"),
        _stint(1, 2, 7, 12, "HARD"),
        _stint(44, 1, 1, 5, "MEDIUM", age=3),
        _stint(44, 2, 6, 12, "HARD"),
    ]
    race_control = [
        {"category": "SafetyCar", "lap_number": 3, "message": "SAFETY CAR DEPLOYED"},
        {"category": "SafetyCar", "lap_number": 4, "message": "SAFETY CAR IN THIS LAP"},
        {"category": "Flag", "lap_number": 9, "message": "YELLOW IN TRACK SECTOR 4"},
    ]
    return build_lap_table("test_race", laps, stints, DRIVERS, race_control)


def _row(table, driver, lap):
    rows = table[(table["Driver"] == driver) & (table["LapNumber"] == lap)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_output_has_exactly_the_tidy_columns(table):
    assert list(table.columns) == TIDY_COLUMNS


def test_used_tyres_continue_from_their_starting_age(table):
    assert _row(table, "HAM", 1)["TyreLife"] == 3
    assert _row(table, "HAM", 3)["TyreLife"] == 5
    assert not _row(table, "HAM", 1)["FittedNew"]


def test_new_set_starts_at_age_zero(table):
    row = _row(table, "VER", 7)
    assert row["TyreLife"] == 0
    assert row["FittedNew"]
    assert row["Compound"] == "HARD"
    assert row["Stint"] == 2


def test_pit_out_lap_and_the_in_lap_before_it_are_flagged(table):
    assert _row(table, "VER", 7)["IsPitOutLap"]
    assert _row(table, "VER", 6)["IsPitInLap"]
    assert not _row(table, "VER", 5)["IsPitInLap"]
    assert not _row(table, "VER", 8)["IsPitOutLap"]


def test_lap_without_a_time_is_dropped(table):
    assert table[(table["Driver"] == "HAM") & (table["LapNumber"] == 10)].empty
    assert not table["LapTimeSec"].isna().any()


def test_safety_car_window_runs_from_the_lap_before_deployment_to_the_restart_lap(table):
    # Race control stamps a message with the lap the leader is about to
    # complete, so the slow laps start one before "DEPLOYED". After a full
    # safety car the restart lap (one after "IN THIS LAP") is slow as well.
    neutralised = sorted(table.loc[table["Driver"] == "VER"].query("Neutralised")["LapNumber"])
    assert neutralised == [2, 3, 4, 5]


def test_virtual_safety_car_ends_on_the_lap_it_is_withdrawn():
    laps = _laps(1, 12)
    stints = [_stint(1, 1, 1, 12, "HARD")]
    race_control = [
        {"category": "SafetyCar", "lap_number": 6, "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"category": "SafetyCar", "lap_number": 7, "message": "VIRTUAL SAFETY CAR ENDING"},
    ]
    table = build_lap_table("r", laps, stints, DRIVERS, race_control)
    assert sorted(table.query("Neutralised")["LapNumber"]) == [5, 6, 7]


def test_red_flag_without_a_safety_car_is_neutralised_through_the_restart():
    laps = _laps(1, 12)
    stints = [_stint(1, 1, 1, 12, "HARD")]
    race_control = [
        {"category": "SessionStatus", "lap_number": 1, "message": "SESSION STARTED"},
        {"category": "Flag", "flag": "RED", "lap_number": 1, "message": "RED FLAG"},
        {"category": "SessionStatus", "lap_number": 2, "message": "SESSION STARTED"},
    ]
    table = build_lap_table("r", laps, stints, DRIVERS, race_control)
    assert sorted(table.query("Neutralised")["LapNumber"]) == [1, 2, 3]


def test_yellow_flag_alone_does_not_neutralise(table):
    assert not _row(table, "VER", 9)["Neutralised"]


def test_total_laps_is_the_longest_lap_count(table):
    assert (table["TotalLaps"] == 12).all()
    assert (table["Race"] == "test_race").all()


def test_safety_car_still_out_at_the_end_neutralises_the_rest():
    laps = _laps(1, 8)
    stints = [_stint(1, 1, 1, 8, "HARD")]
    race_control = [{"category": "SafetyCar", "lap_number": 6, "message": "VIRTUAL SAFETY CAR DEPLOYED"}]
    table = build_lap_table("r", laps, stints, DRIVERS, race_control)
    assert sorted(table.query("Neutralised")["LapNumber"]) == [5, 6, 7, 8]


def test_is_dry_rejects_wet_weather_compounds():
    assert is_dry([_stint(1, 1, 1, 10, "SOFT"), _stint(1, 2, 11, 20, "HARD")])
    assert not is_dry([_stint(1, 1, 1, 10, "SOFT"), _stint(1, 2, 11, 20, "INTERMEDIATE")])
    assert not is_dry([_stint(1, 1, 1, 10, "WET")])


@pytest.mark.parametrize("slug", sorted(RACES))
def test_committed_race_data_is_usable(slug):
    table = load_race(slug)
    assert list(table.columns) == TIDY_COLUMNS
    assert table["Driver"].nunique() >= 15
    assert not table["LapTimeSec"].isna().any()
    assert set(table["Compound"]) <= {"SOFT", "MEDIUM", "HARD"}


def test_red_flag_restart_ends_the_neutralised_period():
    # Safety car, then a red flag: the car is never "called in". Racing
    # resumes with a standing start, which is slow like lap 1.
    laps = _laps(1, 12)
    stints = [_stint(1, 1, 1, 12, "HARD")]
    race_control = [
        {"category": "SessionStatus", "lap_number": 1, "message": "SESSION STARTED"},
        {"category": "SafetyCar", "lap_number": 4, "message": "SAFETY CAR DEPLOYED"},
        {"category": "SessionStatus", "lap_number": 5, "message": "SESSION ABORTED"},
        {"category": "Flag", "flag": "RED", "lap_number": 5, "message": "RED FLAG"},
        {"category": "SessionStatus", "lap_number": 6, "message": "SESSION STARTED"},
    ]
    table = build_lap_table("r", laps, stints, DRIVERS, race_control)
    assert sorted(table.query("Neutralised")["LapNumber"]) == [3, 4, 5, 6, 7]


# Laps that are slow for the whole field but were run under green flags.
# Monaco 2024, lap 3: the first racing lap after the lap-1 red flag, on cold
# tyres and full tanks; 5.3% off the laps around it, just over the threshold.
KNOWN_SLOW_GREEN_LAPS = {"2024_monaco": [3]}


@pytest.mark.parametrize("slug", sorted(RACES))
def test_no_slow_field_lap_is_left_unflagged(slug):
    # If the whole field is slow on a lap, something neutralised the race.
    # Every such lap must carry the flag, so the fit never has to rely on
    # the per-driver outlier filter to remove it.
    table = load_race(slug)
    racing = table[~(table["IsPitInLap"] | table["IsPitOutLap"]) & (table["LapNumber"] > 1)]
    field_median = racing.groupby("LapNumber")["LapTimeSec"].median()
    unflagged = field_median[~field_median.index.isin(table.loc[table["Neutralised"], "LapNumber"])]
    slow = unflagged[unflagged > 1.05 * unflagged.rolling(7, center=True, min_periods=3).median()]
    slow = slow.drop(index=KNOWN_SLOW_GREEN_LAPS.get(slug, []), errors="ignore")
    assert slow.empty, f"slow unflagged laps: {slow.round(1).to_dict()}"


def test_first_timed_lap_after_an_aborted_start_is_treated_as_a_standing_start():
    # A red flag on lap 1 can leave the early laps untimed. The first lap that
    # does have a time is then the standing restart, and is slow like lap 1.
    laps = _laps(1, 10, missing={1, 2})
    stints = [_stint(1, 1, 1, 10, "HARD")]
    table = build_lap_table("r", laps, stints, DRIVERS, [])
    assert sorted(table.query("Neutralised")["LapNumber"]) == [3]
