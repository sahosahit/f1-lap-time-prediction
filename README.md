Overview

This project analyzes the 2024 United States Grand Prix at Circuit of the Americas (COTA) using Formula 1 telemetry data from the FastF1 API.

The goal is to build a race performance modeling pipeline that estimates key factors influencing lap time, including:

driver race pace

tyre degradation

fuel burn effects

strategy implications

The project combines machine learning, statistical modeling, and motorsport performance analysis to simulate race conditions and extract insights similar to those used by F1 performance and strategy engineers.

📊 Key Results
Factor	Estimated Effect
Fuel Burn	0.060 sec / lap improvement
Soft Tyre Degradation	0.066 sec / lap
Medium Tyre Degradation	0.029 sec / lap
Hard Tyre Degradation	0.016 sec / lap
Lap Time Prediction Error	MAE ≈ 0.38 sec

These results align closely with typical Formula 1 race performance models.

🧠 Project Goals

This analysis aims to answer several race engineering questions:

Which drivers had the best race pace?

How quickly do tyres degrade over a stint?

How much lap time improvement comes from fuel burn?

What are the strategy implications for different tyre compounds?

🏎 Project Workflow
Race Data Extraction
        ↓
Feature Engineering
        ↓
Lap Time Prediction (ML Models)
        ↓
Driver Pace Decomposition
        ↓
Fuel Burn Modeling
        ↓
Tyre Degradation Estimation
        ↓
Strategy Insights
📂 Project Structure
f1-lap-time-prediction-cota/
│
├── notebooks/
│   01_data_extraction.ipynb
│   02_feature_engineering.ipynb
│   03_model_training.ipynb
│   04_feature_importance.ipynb
│   05_driver_pace_decomposition.ipynb
│   06_fuel_and_tyre_model.ipynb
│   07_stint_performance_simulator.ipynb
│   08_final_report.ipynb
│
├── data/
│   raw/
│   processed/
│
├── cache/
│
└── README.md
📈 Key Analyses
Driver Race Pace

Driver performance is estimated by removing fuel effects and tyre degradation from lap times.

This allows us to identify true driver pace relative to the field.

Tyre Degradation Modeling

Tyre performance over a stint is estimated using regression models applied to fuel and driver corrected lap times.

Observed degradation rates:

Soft tyres degrade fastest

Medium tyres offer balanced performance

Hard tyres provide stability over longer stints

Fuel Burn Effect

Fuel load decreases throughout the race, making the car progressively faster.

The model estimates:

Fuel burn ≈ 0.060 sec improvement per lap

Over the full race distance, this results in roughly:

≈ 3.3 seconds improvement
Lap Time Prediction

Machine learning models were trained to predict lap time using features such as:

tyre compound

tyre life

fuel load proxy

driver encoding

Models tested:

Linear Regression

Random Forest

Gradient Boosting

Best performance achieved:

Mean Absolute Error ≈ 0.38 sec
🛠 Technologies Used

Python

FastF1

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

🏁 Strategy Insights

Analysis of tyre degradation and fuel burn suggests:

Soft tyres are best suited for short aggressive stints

Medium tyres provide the best compromise between pace and durability

Hard tyres are optimal for longer stints with minimal degradation

These insights are consistent with real-world race strategy decisions.

🚀 Future Work

Potential improvements include:

incorporating traffic and DRS effects

modeling track temperature influence

building a race strategy simulator

adding multi-race datasets for stronger models

📚 Data Source

Race data obtained from the FastF1 API, which provides access to official Formula 1 telemetry and timing data.

https://theoehrly.github.io/Fast-F1/

👨‍💻 Author

Formula 1 enthusiast exploring motorsport data science, race strategy modeling, and performance analytics.

⭐ Why This Project Matters

Modern Formula 1 relies heavily on data-driven performance analysis. This project demonstrates how telemetry data can be used to build models that approximate real race engineering insights.