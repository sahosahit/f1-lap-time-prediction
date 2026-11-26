# F1 Lap Time Prediction and Race Performance Analysis

Two questions about Formula 1 race pace, answered from lap data for the 21 dry races of the 2024 season (the three run on wet or intermediate tyres are left out):

1. **Can lap time be predicted at a race the model has never seen?** Machine-learning models are trained on some races and tested on others, against simple baselines.
2. **What drove lap time within a race?** For any one race, the lap times are split into driver pace, fuel load and tyre wear.

The results section is generated from `results/evaluation.json`, and a test fails if the two disagree.

## Results

<!-- results:start -->
Leave-one-race-out over 21 races and 20,499 clean laps: each race is predicted by a model fitted on the other races.

| Method | Mean absolute error per lap | As a share of lap time |
|---|---|---|
| Gradient boosting **(served by the API)** | 0.59 s | 0.68% |
| Linear (ridge) | 0.62 s | 0.71% |
| Random forest | 0.62 s | 0.71% |
| Baseline: driver's usual pace plus a fuel term | 0.65 s | 0.74% |
| Baseline: each driver at their usual pace | 0.89 s | 1.01% |
| Baseline: every lap equals the race median | 1.02 s | 1.16% |

Per race, mean absolute error in seconds per lap (gradient boosting against the baselines):

| Race | Laps | Model | Driver + fuel | Driver's usual pace | Race median |
|---|---|---|---|---|---|
| Abu Dhabi 2024 | 921 | 0.45 | 0.57 | 0.67 | 0.74 |
| Austin 2024 | 897 | 0.50 | 0.55 | 0.91 | 1.08 |
| Australia 2024 | 837 | 0.51 | 0.59 | 0.80 | 0.89 |
| Austria 2024 | 1257 | 0.46 | 0.57 | 0.57 | 0.69 |
| Azerbaijan 2024 | 880 | 0.61 | 0.64 | 0.89 | 1.02 |
| Bahrain 2024 | 1024 | 0.47 | 0.62 | 0.94 | 1.04 |
| Belgium 2024 | 751 | 0.52 | 0.56 | 1.00 | 1.13 |
| China 2024 | 743 | 0.58 | 0.75 | 0.95 | 1.04 |
| Hungary 2024 | 1255 | 0.60 | 0.72 | 0.91 | 1.04 |
| Imola 2024 | 1164 | 0.62 | 0.69 | 0.60 | 0.86 |
| Japan 2024 | 779 | 0.71 | 0.81 | 1.19 | 1.31 |
| Las Vegas 2024 | 839 | 0.64 | 0.71 | 1.04 | 1.13 |
| Mexico City 2024 | 1045 | 0.54 | 0.62 | 0.89 | 1.00 |
| Miami 2024 | 892 | 0.39 | 0.41 | 0.81 | 0.87 |
| Monaco 2024 | 1170 | 1.18 | 1.04 | 1.40 | 1.60 |
| Monza 2024 | 927 | 0.50 | 0.55 | 0.85 | 1.01 |
| Netherlands 2024 | 1353 | 0.49 | 0.52 | 0.78 | 0.89 |
| Qatar 2024 | 672 | 0.84 | 0.64 | 0.98 | 1.18 |
| Saudi Arabia 2024 | 779 | 0.64 | 0.59 | 0.98 | 1.16 |
| Singapore 2024 | 1109 | 0.65 | 0.76 | 0.85 | 0.93 |
| Spain 2024 | 1205 | 0.58 | 0.65 | 0.93 | 1.08 |
<!-- results:end -->

**What this shows.**

- **Most of the gain is fuel.** Adding a straight-line fuel term to "each driver at their usual pace" removes about a quarter of the error on its own. That baseline knows nothing about tyres.
- **Tyre information adds a little on top.** The best model is about 8% better than the driver-plus-fuel baseline (roughly 0.05 s per lap). That is the honest size of what compound and tyre age contribute here.
- **Gradient boosting is ahead of the linear model and the random forest by about 0.03 s per lap** over the whole set, though the order changes from race to race.
- **About 0.6 s per lap is left unexplained.** That is traffic, track evolution, drivers managing their pace and circuit-specific tyre behaviour, none of which the inputs describe.

## How the prediction works

**Target.** Lap times at different circuits are on different scales (about 85 s at Monza, 100 s at Austin), so the models predict lap time as a percentage of the race's median clean lap. They learn how lap time moves with tyres, fuel and driver, not how long a lap of a circuit is. To turn a prediction into seconds you supply a reference lap time for the race.

**Inputs.** Driver, tyre compound, tyre age (overall and per compound), and the share of the race still to run as a stand-in for fuel load. Drivers and compounds are one-hot encoded: a driver is a category, not a number on a scale. Sector times are not used, because they are parts of the lap time being predicted.

**Which laps.** Pit in-laps and out-laps, lap 1, laps under a safety car or virtual safety car, and laps slower than 107% of the driver's median are left out.

**Evaluation.** Leave-one-race-out: each race is predicted by a model fitted on all the others. The errors in seconds use the held-out race's own median lap as the reference, and its own lap times decide which laps count as outliers, so they are not errors of a forecast made before the race. An earlier version of this project split the laps of a single race at random and reported 0.38 s. That put laps from the same stint on both sides of the split, so the model could score well by memorising the stint; the honest figure for an unseen race is the one in the table.

## What drove lap time within a race

`src/analysis.py` fits one race at a time, all drivers together:

```
lap time = driver pace + compound offset
         + fuel effect × laps remaining
         + wear rate (per compound) × tyre age
```

Fuel and tyre wear have to be estimated together. Fuel burns off and tyres wear at the same time, so fitting lap time against lap number alone mixes the two up. Fitting the whole field at once separates them, because drivers pit on different laps. A compound needs at least 40 clean laps in the race to be fitted and a driver at least 10; otherwise they are reported as not fitted.

The fit can return values that are not physical, and it says so when it does. At Qatar 2024 both wear rates come out slightly negative: lap times fell as the tyres aged, which means the track getting faster outweighed the wear. The API returns a warning with those numbers. The compound offsets are also relative to whichever compound was used most in that race, and teams choose when to run each compound, so they are not a controlled comparison of the tyres.

Example, United States Grand Prix (Austin) 2024:

| | |
|---|---|
| Fuel effect | 0.069 s per lap of fuel |
| Hard tyre wear | 0.036 s per lap (527 laps) |
| Medium tyre wear | 0.044 s per lap (367 laps) |
| Soft tyre | not fitted (3 clean laps) |
| Fastest race pace | LEC −1.23 s, SAI −1.10 s, NOR −1.01 s, PIA −0.90 s vs the field average |

These come from `GET /analysis/2024_austin` and can be reproduced with `fit_race_effects`.

## Limitations

- **One season, dry races only.** Driver effects are really driver-and-car effects for 2024 and would not carry to another season.
- **The reference lap time comes from the race itself.** In the evaluation it is the held-out race's own median, so the models are tested on the shape of lap time through a race, not on predicting a circuit's lap time in advance.
- **No traffic, weather or track temperature**, and no information about which circuit it is.
- **Tyre wear is linear in the per-race analysis**, with no cliff, and it is not separated from the track getting faster as rubber goes down.
- **Laps within a stint are not independent**, so the per-lap errors understate how far a whole stint prediction can drift.

## Running it

```bash
pip install -r requirements.txt

pytest                         # run the tests
python src/train.py            # evaluate, update results and README, save the best model
python src/openf1_loader.py    # re-download the race data (not needed; it is committed)
uvicorn api.main:app --reload  # or: docker compose up
```

| Endpoint | What it does |
|---|---|
| `GET /health`, `/races`, `/drivers` | What is loaded |
| `POST /predict` | Lap time for a driver, compound, tyre age and point in the race |
| `GET /analysis/{race}` | Fuel effect, tyre wear and driver pace for one race, with warnings |
| `POST /predict/stint` | Expected lap times for a stint in a given race |

The service loads the saved model, or fits one from the committed data on start-up if none has been saved.

## Project structure

```
f1-lap-time-prediction/
├── src/
│   ├── openf1_loader.py        # OpenF1 data → one tidy lap table per race
│   ├── races.py                # the races used
│   ├── feature_engineering.py  # clean laps, build the modelling dataset
│   ├── model.py                # models, baselines, leave-one-race-out evaluation
│   ├── train.py                # run the evaluation, write results, save the model
│   ├── analysis.py             # per-race fuel, tyre and pace fit
│   └── preprocessing.py, data_loader.py   # FastF1 helpers used by the notebooks
├── api/main.py                 # FastAPI service
├── data/races/                 # tidy lap tables, one CSV per race
├── results/evaluation.json     # numbers of record
├── notebooks/                  # original single-race exploration (see below)
└── tests/
```

## Notebooks

The notebooks are the original exploration of one race (Austin 2024) that this project grew out of. Their printed numbers predate the method described above: they use a random split within one race, a driver ID treated as a number, and separate fuel and tyre fits. They are kept for the analysis ideas; the figures of record are in the results section.

## Data

Race data comes from the [OpenF1 API](https://openf1.org) (laps, tyre stints and race-control messages) and is stored in `data/races/`. The notebook data in `data/raw/` and `data/processed/` came from [FastF1](https://docs.fastf1.dev).

## Future work

- Circuit features (length, corner mix, tyre allocation) so the level of lap time can be predicted, not only its shape.
- More seasons, with car and driver effects separated.
- Non-linear tyre wear and track evolution in the per-race fit.
- Track temperature and traffic.
