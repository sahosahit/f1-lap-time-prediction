# F1 Lap Time Prediction and Race Performance Analysis

Two questions about Formula 1 race pace, answered from lap data for sixteen dry races of the 2024 season:

1. **Can lap time be predicted at a race the model has never seen?** Machine-learning models are trained on some races and tested on others, against simple baselines.
2. **What drove lap time within a race?** For any one race, the lap times are split into driver pace, fuel load and tyre wear.

The results section is generated from `results/evaluation.json`, and a test fails if the two disagree.

## Results

<!-- results:start -->
Leave-one-race-out over 16 races and 15,861 clean laps: each race is predicted by a model fitted on the other races.

| Method | Mean absolute error per lap | As a share of lap time |
|---|---|---|
| Gradient boosting **(best)** | 0.57 s | 0.64% |
| Linear (ridge) | 0.57 s | 0.64% |
| Random forest | 0.59 s | 0.67% |
| Baseline: each driver at their usual pace | 0.88 s | 0.99% |
| Baseline: Every lap equals the race median | 1.00 s | 1.13% |

Per race, mean absolute error in seconds per lap (gradient boosting against the two baselines):

| Race | Laps | Model | Driver's usual pace | Race median |
|---|---|---|---|---|
| Abu Dhabi 2024 | 921 | 0.46 | 0.67 | 0.74 |
| Austin 2024 | 916 | 0.55 | 0.95 | 1.12 |
| Austria 2024 | 1257 | 0.45 | 0.58 | 0.69 |
| Azerbaijan 2024 | 880 | 0.60 | 0.89 | 1.02 |
| Bahrain 2024 | 1024 | 0.46 | 0.94 | 1.04 |
| China 2024 | 761 | 0.57 | 0.94 | 1.04 |
| Hungary 2024 | 1255 | 0.61 | 0.92 | 1.04 |
| Japan 2024 | 779 | 0.72 | 1.19 | 1.31 |
| Mexico City 2024 | 1063 | 0.53 | 0.89 | 1.00 |
| Miami 2024 | 911 | 0.41 | 0.81 | 0.87 |
| Monza 2024 | 927 | 0.51 | 0.85 | 1.01 |
| Netherlands 2024 | 1353 | 0.50 | 0.79 | 0.89 |
| Qatar 2024 | 703 | 0.97 | 1.06 | 1.25 |
| Saudi Arabia 2024 | 797 | 0.70 | 1.03 | 1.22 |
| Singapore 2024 | 1109 | 0.64 | 0.84 | 0.93 |
| Spain 2024 | 1205 | 0.58 | 0.93 | 1.08 |
<!-- results:end -->

**What this shows.**

- Knowing the tyre, its age and the fuel load cuts the error by roughly a third compared with knowing only who is driving.
- The three models are close. A plain linear model is within a hundredth of a second of gradient boosting, so most of what can be learned from these inputs is simple: wear and fuel act roughly linearly.
- About half a second per lap is left unexplained. That is traffic, track evolution, driver management of pace and circuit-specific tyre behaviour, none of which the inputs describe.

## How the prediction works

**Target.** Lap times at different circuits are on different scales (about 80 s at Monza, 100 s at Austin), so the models predict lap time as a percentage of the race's median clean lap. They learn how lap time moves with tyres, fuel and driver, not how long a lap of a circuit is. To turn a prediction into seconds you supply a reference lap time for the race.

**Inputs.** Driver, tyre compound, tyre age (overall and per compound), and the share of the race still to run as a stand-in for fuel load. Drivers and compounds are one-hot encoded: a driver is a category, not a number on a scale. Sector times are not used, because they are parts of the lap time being predicted.

**Which laps.** Pit in-laps and out-laps, lap 1, laps under a safety car or virtual safety car, and laps slower than 107% of the driver's median are left out.

**Evaluation.** Leave-one-race-out: each race is predicted by a model fitted on all the others. An earlier version of this project split the laps of a single race at random and reported 0.38 s. That put laps from the same stint on both sides of the split, so the model could score well by memorising the stint; the honest figure for an unseen race is the one in the table.

## What drove lap time within a race

`src/analysis.py` fits one race at a time, all drivers together:

```
lap time = driver pace + compound offset
         + fuel effect × laps remaining
         + wear rate (per compound) × tyre age
```

Fuel and tyre wear have to be estimated together. Fuel burns off and tyres wear at the same time, so fitting lap time against lap number alone mixes the two up. Fitting the whole field at once separates them, because drivers pit on different laps. A compound needs at least 40 clean laps in the race to be fitted; otherwise it is reported as not fitted.

Example, United States Grand Prix (Austin) 2024:

| | |
|---|---|
| Fuel effect | 0.072 s per lap of fuel |
| Hard tyre wear | 0.036 s per lap (531 laps) |
| Medium tyre wear | 0.038 s per lap (382 laps) |
| Soft tyre | not fitted (3 clean laps) |
| Fastest race pace | LEC −1.27 s, SAI −1.14 s, NOR −1.01 s, VER −0.91 s vs the field average |

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
| `GET /analysis/{race}` | Fuel effect, tyre wear and driver pace for one race |
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
