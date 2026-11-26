"""
Races used for training and evaluation: slug -> OpenF1 session key.

Every 2024 Grand Prix is listed. A race where intermediate or wet tyres were
used is skipped when the data is downloaded (the features only describe dry
running), so it has no file in data/races/ and is left out of DRY_RACES.
"""

import os

ALL_RACES = {
    "2024_bahrain": 9472,
    "2024_saudi_arabia": 9480,
    "2024_australia": 9488,
    "2024_japan": 9496,
    "2024_china": 9673,
    "2024_miami": 9507,
    "2024_imola": 9515,
    "2024_monaco": 9523,
    "2024_canada": 9531,
    "2024_spain": 9539,
    "2024_austria": 9550,
    "2024_silverstone": 9558,
    "2024_hungary": 9566,
    "2024_belgium": 9574,
    "2024_netherlands": 9582,
    "2024_monza": 9590,
    "2024_azerbaijan": 9598,
    "2024_singapore": 9606,
    "2024_austin": 9617,
    "2024_mexico_city": 9625,
    "2024_brazil": 9636,
    "2024_las_vegas": 9644,
    "2024_qatar": 9655,
    "2024_abu_dhabi": 9662,
}

_RACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "races")

# The races that have data: every dry race that has been downloaded.
RACES = {
    slug: key for slug, key in ALL_RACES.items()
    if os.path.exists(os.path.join(_RACES_DIR, f"{slug}.csv"))
}
