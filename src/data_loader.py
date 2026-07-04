"""
FastF1 data loading for race lap data.
"""

import os
import fastf1
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_laps(year, race_name, session='R'):
    """
    Load lap data for a specific race from FastF1 API.

    Args:
        year: Season year (e.g., 2024)
        race_name: Race name or round number
        session: Session type ('R' for race, 'Q' for qualifying)

    Returns:
        DataFrame with lap timing data
    """
    session_obj = fastf1.get_session(year, race_name, session)
    session_obj.load(telemetry=False, weather=False, messages=False)
    return session_obj.laps.copy()
