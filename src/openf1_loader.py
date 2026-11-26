"""
Race lap data from the OpenF1 API (https://openf1.org).

Builds one tidy table per race: one row per timed lap, with the tyre the
driver was on, how old it was, and whether the lap is usable for modelling
(pit laps and safety-car laps are flagged, not removed).

Fetched tables are stored as CSV under data/races/ so that tests and CI
never need the network.

    python src/openf1_loader.py     # download every race in races.py
"""

import os
import time

import pandas as pd
import requests

from races import ALL_RACES, RACES

API_URL = "https://api.openf1.org/v1"
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
REQUEST_INTERVAL_SEC = 2.5
RETRY_WAITS_SEC = (5, 30)

TIDY_COLUMNS = [
    "Race", "Driver", "LapNumber", "LapTimeSec", "Stint", "Compound",
    "TyreLife", "FittedNew", "IsPitInLap", "IsPitOutLap", "Neutralised",
    "TotalLaps",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACES_DIR = os.path.join(REPO_ROOT, "data", "races")


def is_dry(stints):
    """True if no driver used intermediate or wet tyres."""
    return not any(s.get("compound") in WET_COMPOUNDS for s in stints)


def neutralised_laps(race_control, total_laps):
    """
    Race laps run under a safety car, virtual safety car or red flag.

    Race control stamps each message with the lap the leader is about to
    complete, which runs one ahead of the lap being timed. So the slow laps
    start one lap before the "DEPLOYED" (or red flag) message. They end:
    - on the lap stamped "ENDING" for a virtual safety car;
    - one lap after "IN THIS LAP" for a full safety car, because the field
      restarts bunched up and that lap is slow too;
    - on the standing restart lap after a red flag.
    A period that is never closed covers the rest of the race.
    """
    def is_red_flag(message):
        return message.get("category") == "Flag" and message.get("flag") == "RED"

    events = sorted(
        (m for m in race_control
         if (m.get("category") in ("SafetyCar", "SessionStatus") or is_red_flag(m))
         and m.get("lap_number") is not None),
        key=lambda m: m["lap_number"],
    )
    laps = set()
    open_from = None
    for message in events:
        text = (message.get("message") or "").upper()
        category, lap = message["category"], message["lap_number"]
        if open_from is None:
            if (category == "SafetyCar" and "DEPLOYED" in text) or is_red_flag(message):
                open_from = lap - 1
        elif category == "SafetyCar" and "IN THIS LAP" in text:
            laps.update(range(open_from, lap + 2))
            open_from = None
        elif category == "SafetyCar" and "ENDING" in text:
            laps.update(range(open_from, lap + 1))
            open_from = None
        elif category == "SessionStatus" and "STARTED" in text:
            laps.update(range(open_from, lap + 2))
            open_from = None
    if open_from is not None:
        laps.update(range(open_from, total_laps + 1))
    return {lap for lap in laps if lap >= 1}


def first_full_field_lap(table):
    """The first lap number that at least half of the drivers have a time for."""
    if table.empty:
        return 1
    timed = table.groupby("LapNumber")["Driver"].nunique()
    return int(timed[timed >= table["Driver"].nunique() / 2].index.min())


def build_lap_table(slug, laps, stints, drivers, race_control):
    """Combine raw OpenF1 records into the tidy lap table. No network access."""
    acronym = {d["driver_number"]: d["name_acronym"] for d in drivers}
    total_laps = max(lap["lap_number"] for lap in laps)
    neutralised = neutralised_laps(race_control, total_laps)
    pit_out = {
        (lap["driver_number"], lap["lap_number"])
        for lap in laps if lap.get("is_pit_out_lap")
    }

    rows = []
    for lap in laps:
        number, lap_number = lap["driver_number"], lap["lap_number"]
        stint = next(
            (s for s in stints
             if s["driver_number"] == number
             and s["lap_start"] <= lap_number <= s["lap_end"]),
            None,
        )
        if (lap.get("lap_duration") is None or stint is None
                or stint.get("compound") not in DRY_COMPOUNDS
                or number not in acronym):
            continue
        age_at_start = stint.get("tyre_age_at_start") or 0
        rows.append({
            "Race": slug,
            "Driver": acronym[number],
            "LapNumber": int(lap_number),
            "LapTimeSec": float(lap["lap_duration"]),
            "Stint": int(stint["stint_number"]),
            "Compound": stint["compound"],
            "TyreLife": int(age_at_start + lap_number - stint["lap_start"]),
            "FittedNew": age_at_start == 0,
            "IsPitInLap": (number, lap_number + 1) in pit_out,
            "IsPitOutLap": (number, lap_number) in pit_out,
            "Neutralised": lap_number in neutralised,
            "TotalLaps": int(total_laps),
        })

    table = pd.DataFrame(rows, columns=TIDY_COLUMNS)
    # If the first lap that most of the field has a time for is not lap 1 (an
    # aborted start left the early laps untimed), that lap is the standing
    # restart: slow like lap 1.
    restart_lap = first_full_field_lap(table)
    if restart_lap > 1:
        table.loc[table["LapNumber"] <= restart_lap, "Neutralised"] = True
    return table.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def _get(endpoint, session_key):
    """
    GET one OpenF1 endpoint for a session.

    The free API allows about 30 requests a minute, so calls are spaced
    out, and a busy response (429 or 5xx) is retried after a longer wait.
    """
    url = f"{API_URL}/{endpoint}"
    for wait in RETRY_WAITS_SEC + (None,):
        time.sleep(REQUEST_INTERVAL_SEC)
        response = requests.get(url, params={"session_key": session_key}, timeout=60)
        busy = response.status_code == 429 or response.status_code >= 500
        if busy and wait is not None:
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()


def fetch_race(slug):
    """Download one race from OpenF1 and return its tidy lap table."""
    session_key = ALL_RACES[slug]
    stints = _get("stints", session_key)
    if not is_dry(stints):
        raise ValueError(f"{slug} is not a dry race")
    return build_lap_table(
        slug,
        _get("laps", session_key),
        stints,
        _get("drivers", session_key),
        _get("race_control", session_key),
    )


def race_path(slug):
    return os.path.join(RACES_DIR, f"{slug}.csv")


def save_race(table, slug):
    os.makedirs(RACES_DIR, exist_ok=True)
    table.to_csv(race_path(slug), index=False)


def load_race(slug):
    """Read a previously fetched race from data/races/."""
    return pd.read_csv(race_path(slug))


def main():
    for slug in ALL_RACES:
        try:
            table = fetch_race(slug)
        except ValueError as error:
            print(f"SKIPPED {slug}: {error}")
            continue
        save_race(table, slug)
        print(f"{slug}: {len(table)} laps, {table['Driver'].nunique()} drivers, "
              f"{table['Compound'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
