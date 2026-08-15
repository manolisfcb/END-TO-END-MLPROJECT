# Traffic Congestion Forecasting Project — Portfolio Roadmap

## Project idea

Build a machine learning system that predicts whether congestion will form on a road segment / bridge-like traffic sensor in the next **30, 60, 90, and 120 minutes**.

The final goal is to simulate a traffic operations early-warning system:

> Given current traffic conditions, recent traffic history, calendar context, incidents, weather, and possibly nearby sensors, estimate the probability of congestion forming soon so operators can take preventive actions.

---

## Current project context

Dataset: **XTraffic / TraffiDent**

Local dataset structure observed:

```text
archive/
├── year_2022/
├── year_2023/
├── year_2024/
├── incidents_y2022.csv
├── incidents_y2023.csv
├── incidents_y2024.csv
├── sensor_meta_feature.csv
├── node_order.npy
├── adj_matrix.npy
└── dis_matrix.npy
```

Main traffic file tested:

```text
year_2024/year_2024/2024_p01.npy
```

Observed shape:

```python
traffic.shape == (16972, 8928, 3)
```

Interpretation:

```text
16972 sensors/nodes
8928 timestamps for January 2024
3 traffic variables per timestamp: [flow, occupancy, speed]
```

Important access pattern:

```python
traffic[sensor_index, timestamp_index, variable_index]
```

The file does **not** appear to include explicit timestamps. For January 2024, timestamps were reconstructed because:

```text
31 days × 24 hours × 12 measurements/hour = 8928 measurements
```

So the working assumption is:

```text
frequency = 5 minutes
2024_p01 = January 2024
```

This assumption should be validated against all monthly file shapes.

---

## Selected MVP sensor

A candidate sensor was selected based on speed variation and congestion behavior.

Current candidate:

```text
sensor_id = 819086
month = January 2024
```

The sensor showed realistic congestion behavior on:

```text
2024-01-26
```

Observed daily summary for that day:

```text
avg_speed = 35.198264
min_speed = 8.4
speed_std = 22.601562
slow_rate_30 = 0.562500
flow_mean = 199.996528
occupancy_mean = 0.221065
```

Interpretation:

```text
The sensor was below speed 30 for 56.25% of the day.
```

Since there are 288 measurements per day:

```text
0.5625 × 288 = 162 intervals
162 intervals × 5 minutes = 810 minutes
```

So this day had about **13.5 hours** of speed below 30.

---

## Congestion definition

Initial congestion label:

```python
df_selected["is_congested"] = (df_selected["speed"] < 30).astype(int)
```

This is a first operational definition, not necessarily the final one.

Potential future experiments:

```text
speed < 25
speed < 30
speed < 35
speed < 40
```

Current working threshold:

```text
speed < 30
```

Reason: visually, speed below 30 separated normal traffic from sustained low-speed congestion for the selected sensor.

---

## Observed congestion blocks on 2024-01-26

Main sustained congestion blocks:

```text
06:10 → 10:45
Duration: 280 minutes
Avg speed: 15.155357
Min speed: 8.4
Avg occupancy: 0.347437
Avg flow: 194.678571
```

```text
11:20 → 17:50
Duration: 395 minutes
Avg speed: 14.968354
Min speed: 10.1
Avg occupancy: 0.351510
Avg flow: 221.164557
```

```text
18:00 → 19:20
Duration: 85 minutes
Avg speed: 23.770588
Min speed: 20.0
Avg occupancy: 0.265241
Avg flow: 275.176471
```

Smaller blocks:

```text
19:55 → 20:30
Duration: 40 minutes
Avg speed: 28.637500
```

```text
19:35 → 19:35
Duration: 5 minutes
Avg speed: 29.900000
```

```text
19:45 → 19:45
Duration: 5 minutes
Avg speed: 29.500000
```

Conclusion:

```text
This appears to be real congestion behavior, not random noise.
Speed drops for long blocks, occupancy increases when speed decreases, and flow behaves plausibly.
```

---

# Project stages and checklist

## Stage 1 — Understand one sensor and one month

Goal: understand the dataset structure and validate that one selected sensor/month is usable.

### Dataset structure

- [x] Download XTraffic / TraffiDent dataset.
- [x] Inspect root files and folders.
- [x] Identify yearly traffic folders: `year_2022`, `year_2023`, `year_2024`.
- [x] Identify incident files: `incidents_y2022.csv`, `incidents_y2023.csv`, `incidents_y2024.csv`.
- [x] Identify sensor metadata file: `sensor_meta_feature.csv`.
- [x] Identify network files: `node_order.npy`, `adj_matrix.npy`, `dis_matrix.npy`.

### Understand node order

- [x] Load `node_order.npy`.
- [x] Confirm shape: `(16972,)`.
- [x] Understand that `node_order[index]` maps matrix position to real `sensor_id`.
- [x] Understand that `traffic[0]` corresponds to sensor `node_order[0]`.

### Understand traffic matrix

- [x] Load one monthly traffic file: `2024_p01.npy`.
- [x] Confirm shape: `(16972, 8928, 3)`.
- [x] Interpret dimensions as `sensor × time × variable`.
- [x] Understand that variable order is assumed to be `[flow, occupancy, speed]`.
- [ ] Validate variable order from official metadata or documentation.
- [x] Reconstruct timestamp for January 2024 using 5-minute frequency.
- [ ] Validate all monthly file shapes to confirm `p01 = January`, `p02 = February`, etc.

### Select a useful sensor

- [x] Create summary statistics across sensors.
- [x] Calculate `speed_mean`, `speed_std`, `speed_min`, `slow_rate_30`, `zero_speed_rate`.
- [x] Filter for sensors with meaningful variation and low missing/zero issues.
- [x] Select candidate sensor: `sensor_id = 819086`.
- [ ] Join selected sensor with `sensor_meta_feature.csv` to get freeway, city, direction, lat/lng.

### Visual EDA

- [x] Create `df_selected` for selected sensor and January 2024.
- [x] Plot speed for worst day: `2024-01-26`.
- [x] Plot occupancy for worst day.
- [x] Plot flow for worst day.
- [x] Confirm speed drops by long blocks.
- [x] Confirm occupancy rises when speed drops.
- [x] Conclude this looks like real congestion, not random data noise.

### Define congestion

- [x] Define initial congestion label: `is_congested = speed < 30`.
- [x] Calculate daily congestion rate.
- [x] Identify sustained congestion blocks.
- [ ] Later test alternative thresholds: 25, 35, 40.

---

## Stage 2 — Create future targets and baseline model for one sensor / one month

Goal: transform the problem into multi-horizon classification.

### Future targets

Current horizons:

```python
horizons = {
    "30min": 6,
    "60min": 12,
    "90min": 18,
    "120min": 24
}
```

Because:

```text
1 step = 5 minutes
6 steps = 30 minutes
12 steps = 60 minutes
18 steps = 90 minutes
24 steps = 120 minutes
```

Targets created:

```python
df_model["congestion_30min"] = df_model["is_congested"].shift(-6)
df_model["congestion_60min"] = df_model["is_congested"].shift(-12)
df_model["congestion_90min"] = df_model["is_congested"].shift(-18)
df_model["congestion_120min"] = df_model["is_congested"].shift(-24)
```

Checklist:

- [x] Create `congestion_30min`.
- [x] Create `congestion_60min`.
- [x] Create `congestion_90min`.
- [x] Create `congestion_120min`.
- [x] Validate that future targets are correctly aligned.
- [x] Check target values around first congestion block.
- [ ] Calculate target balance for all horizons.

### Feature engineering

Goal: give the model only information available at the current timestamp or earlier.

Calendar features:

- [x] `hour`
- [x] `minute`
- [x] `day_of_week`
- [x] `day`
- [x] `is_weekend`
- [x] `hour_sin`
- [x] `hour_cos`
- [x] `dow_sin`
- [x] `dow_cos`

Lag features used for MVP:

```python
lag_steps = {
    "5min": 1,
    "15min": 3,
    "30min": 6,
    "60min": 12
}
```

Reason:

```text
For the first MVP, use recent history up to 60 minutes to capture short-term traffic dynamics.
Longer lags such as 90/120 minutes can be tested later as an experiment.
```

Lag checklist:

- [x] Create speed lags up to 60 minutes.
- [x] Create flow lags up to 60 minutes.
- [x] Create occupancy lags up to 60 minutes.
- [x] Create speed differences: 5, 15, 30 minutes.
- [x] Create occupancy differences: 5, 15 minutes.

Rolling features:

```python
rolling_windows = {
    "15min": 3,
    "30min": 6,
    "60min": 12
}
```

Rolling checklist:

- [x] Create rolling mean for speed.
- [x] Create rolling std for speed.
- [x] Create rolling mean for flow.
- [x] Create rolling mean for occupancy.
- [x] Use `.shift(1)` before rolling calculations to avoid leakage.
- [x] Drop rows with NaN caused by lags/rolling windows.
- [ ] Inspect final feature matrix shape.
- [ ] Inspect missing values after feature engineering.

### Baseline model

Next step.

Baseline type:

```text
Persistence baseline
```

Logic:

```text
If the sensor is congested now, predict it will be congested in the future.
If the sensor is not congested now, predict it will not be congested in the future.
```

In code:

```python
y_pred = test_df["is_congested"]
```

Checklist:

- [ ] Sort data by timestamp.
- [ ] Create temporal train/test split.
- [ ] Use first 70% of time for training and last 30% for testing.
- [ ] Evaluate persistence baseline for `congestion_30min`.
- [ ] Evaluate persistence baseline for `congestion_60min`.
- [ ] Evaluate persistence baseline for `congestion_90min`.
- [ ] Evaluate persistence baseline for `congestion_120min`.
- [ ] Calculate accuracy, precision, recall, and F1.
- [ ] Inspect confusion matrix for 30-minute horizon.
- [ ] Evaluate early-warning subset where `is_congested == 0` now.
- [ ] Save baseline results as CSV.

---

## Stage 3 — First ML models for one sensor / one month

Goal: check whether a real ML model improves over the persistence baseline.

Models:

- [ ] Logistic Regression for each horizon.
- [ ] Random Forest for each horizon.
- [ ] XGBoost / LightGBM for each horizon.

Evaluation:

- [ ] Use same temporal split as baseline.
- [ ] Compare against persistence baseline.
- [ ] Evaluate all horizons separately.
- [ ] Focus on recall and F1, not only accuracy.
- [ ] Pay special attention to false negatives.
- [ ] Evaluate early-warning cases separately.

Expected result:

```text
The model should be especially better than baseline when traffic is not congested now but will become congested soon.
```

---

## Stage 4 — Scale from one month to all months of 2024

Goal: turn the current notebook logic into reusable functions.

Required functions:

- [ ] `load_traffic_month(year, month)`
- [ ] `load_sensor_month(year, month, sensor_index)`
- [ ] `build_timestamp_index(year, month, n_rows)`
- [ ] `create_congestion_label(df, threshold=30)`
- [ ] `create_future_targets(df, horizons)`
- [ ] `create_time_features(df)`
- [ ] `create_lag_features(df, lag_steps)`
- [ ] `create_rolling_features(df, rolling_windows)`
- [ ] `build_feature_matrix(df)`

Then:

- [ ] Load all months of 2024 for selected sensor.
- [ ] Concatenate all months into one DataFrame.
- [ ] Rebuild targets across month boundaries carefully.
- [ ] Rebuild features across month boundaries carefully.
- [ ] Train/test split by date.
- [ ] Re-evaluate baseline and ML models.

---

## Stage 5 — Scale to 2022, 2023, and 2024

Goal: create a multi-year dataset for the selected sensor.

Possible split:

```text
Train: 2022 + 2023
Test: 2024
```

Checklist:

- [ ] Load selected sensor data for all months in 2022.
- [ ] Load selected sensor data for all months in 2023.
- [ ] Load selected sensor data for all months in 2024.
- [ ] Concatenate into a multi-year DataFrame.
- [ ] Validate timestamp continuity.
- [ ] Check missing values and zero rates by year.
- [ ] Recompute targets and features.
- [ ] Train on 2022–2023.
- [ ] Test on 2024.
- [ ] Compare performance against one-month model.

---

## Stage 6 — Add incident reports

Goal: enrich the model with external traffic incident information.

Incident files:

```text
incidents_y2022.csv
incidents_y2023.csv
incidents_y2024.csv
```

Potential incident features:

- [ ] `incident_active_nearby`
- [ ] `incident_count_last_30min`
- [ ] `incident_count_last_60min`
- [ ] `incident_type`
- [ ] `minutes_since_last_incident`
- [ ] `distance_to_nearest_incident`
- [ ] `same_freeway_incident`
- [ ] `same_direction_incident`

Steps:

- [ ] Load incident files.
- [ ] Parse incident datetime column.
- [ ] Understand incident types.
- [ ] Join incidents to sensor using time proximity.
- [ ] Join incidents to sensor using spatial proximity or freeway/direction.
- [ ] Create incident features.
- [ ] Train model with and without incident features.
- [ ] Measure whether incidents improve early-warning performance.

---

## Stage 7 — Add calendar and holiday features

Goal: give the model additional context about special days.

Features:

- [ ] `is_holiday`
- [ ] `is_day_before_holiday`
- [ ] `is_day_after_holiday`
- [ ] `is_rush_hour_morning`
- [ ] `is_rush_hour_evening`
- [ ] `is_business_day`

Experiments:

- [ ] Traffic-only model.
- [ ] Traffic + calendar model.
- [ ] Compare performance.

---

## Stage 8 — Add weather data

Goal: test whether weather improves congestion prediction.

Potential weather features:

- [ ] temperature
- [ ] precipitation
- [ ] rain indicator
- [ ] snow indicator
- [ ] visibility
- [ ] wind speed
- [ ] weather severity index

Steps:

- [ ] Identify weather data source.
- [ ] Match weather station/location to selected sensor.
- [ ] Load historical weather data.
- [ ] Resample weather to 5-minute or hourly frequency.
- [ ] Join weather with traffic by timestamp.
- [ ] Train model with and without weather.
- [ ] Measure improvement.

---

## Stage 9 — Add nearby sensors / spatial context

Goal: use upstream/downstream traffic conditions to predict congestion earlier.

Network files:

```text
adj_matrix.npy
```

```text
dis_matrix.npy
```

Potential features:

- [ ] upstream sensor speed
- [ ] upstream sensor occupancy
- [ ] downstream sensor speed
- [ ] downstream sensor occupancy
- [ ] average speed of nearest sensors
- [ ] congestion count among nearest sensors

Steps:

- [ ] Understand `adj_matrix.npy`.
- [ ] Understand `dis_matrix.npy`.
- [ ] Find nearest or connected sensors to selected sensor.
- [ ] Load traffic data for nearby sensors.
- [ ] Create spatial features.
- [ ] Train model with and without spatial features.
- [ ] Measure whether nearby sensors improve early warning.

---

## Stage 10 — Model comparison and final evaluation

Goal: produce a strong portfolio-ready evaluation.

Model groups:

- [ ] Persistence baseline.
- [ ] Logistic Regression.
- [ ] Random Forest.
- [ ] XGBoost / LightGBM.
- [ ] Optional: LSTM.
- [ ] Optional: graph-based model.

Comparison table:

```text
Model                         30min   60min   90min   120min
Persistence baseline           ...     ...     ...      ...
Logistic Regression            ...     ...     ...      ...
Random Forest                  ...     ...     ...      ...
XGBoost                        ...     ...     ...      ...
XGBoost + incidents            ...     ...     ...      ...
XGBoost + weather              ...     ...     ...      ...
XGBoost + spatial features     ...     ...     ...      ...
```

Metrics:

- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1.
- [ ] ROC-AUC if probability outputs are used.
- [ ] PR-AUC if classes are imbalanced.
- [ ] Confusion matrix by horizon.
- [ ] Early-warning recall.

---

## Stage 11 — Turn notebook into project structure

Goal: make the project look professional in GitHub.

Target structure:

```text
traffic-congestion-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_model_comparison.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── models/
├── reports/
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

Checklist:

- [ ] Move repeated notebook logic into Python modules.
- [ ] Add clear README.
- [ ] Add project diagram.
- [ ] Add sample plots.
- [ ] Add model comparison table.
- [ ] Add instructions to reproduce.
- [ ] Add portfolio summary.

---

## Stage 12 — Optional MLOps / API layer

Goal: make the project look like an ML engineering system, not only a notebook.

Possible additions:

- [ ] FastAPI endpoint for prediction.
- [ ] Dockerfile.
- [ ] MLflow experiment tracking.
- [ ] Model artifact saved as `.pkl` or `.joblib`.
- [ ] Batch prediction script.
- [ ] Simple dashboard.
- [ ] Data validation checks.
- [ ] Basic tests.

Example API behavior:

```text
Input:
current speed, flow, occupancy, recent lags, calendar features, incident features

Output:
P(congestion in 30min)
P(congestion in 60min)
P(congestion in 90min)
P(congestion in 120min)
```

---

# Useful code snippets already used

## Load traffic month

```python
import numpy as np
from pathlib import Path

BASE_PATH = Path("../Data/archive")

traffic = np.load(
    BASE_PATH / "year_2024/year_2024/2024_p01.npy",
    mmap_mode="r"
)

print(traffic.shape)
```

## Load node order

```python
node_order = np.load(
    BASE_PATH / "node_order.npy",
    allow_pickle=True
)

print(node_order.shape)
print(node_order[:20])
```

## Create selected sensor DataFrame

```python
import pandas as pd

selected_data = traffic[selected_sensor_index, :, :]

city_sensor_df = pd.DataFrame(
    selected_data,
    columns=["flow", "occupancy", "speed"]
)

city_sensor_df["timestamp"] = pd.date_range(
    start="2024-01-01",
    periods=len(city_sensor_df),
    freq="5min"
)

city_sensor_df["sensor_index"] = selected_sensor_index
city_sensor_df["sensor_id"] = selected_sensor_id
```

## Create congestion label

```python
df_selected["is_congested"] = (df_selected["speed"] < 30).astype(int)
```

## Create future targets

```python
horizons = {
    "30min": 6,
    "60min": 12,
    "90min": 18,
    "120min": 24
}

for horizon_name, steps in horizons.items():
    df_model[f"congestion_{horizon_name}"] = (
        df_model["is_congested"].shift(-steps)
    )
```

## Create lag features

```python
lag_steps = {
    "5min": 1,
    "15min": 3,
    "30min": 6,
    "60min": 12
}

for lag_name, steps in lag_steps.items():
    df_model[f"speed_lag_{lag_name}"] = df_model["speed"].shift(steps)
    df_model[f"flow_lag_{lag_name}"] = df_model["flow"].shift(steps)
    df_model[f"occupancy_lag_{lag_name}"] = df_model["occupancy"].shift(steps)
```

---

# Prompt to continue in a new ChatGPT conversation

Use this prompt if the current chat becomes too long:

```text
I am building a portfolio project called Traffic Congestion Forecasting using the XTraffic / TraffiDent dataset.

The goal is to predict whether congestion will occur in the next 30, 60, 90, and 120 minutes for a selected road sensor, using traffic variables, calendar features, lags, rolling features, and later incidents, weather, and nearby sensors.

Current dataset understanding:
- Traffic files are `.npy` arrays.
- Example file: `year_2024/year_2024/2024_p01.npy`.
- Observed shape: `(16972, 8928, 3)`.
- Interpretation: `sensor × timestamp × variable`.
- Variables are assumed to be `[flow, occupancy, speed]`.
- `node_order.npy` maps sensor index to real `sensor_id`.
- January 2024 has 8928 measurements, consistent with 5-minute frequency.

Current selected sensor:
- `sensor_id = 819086`.
- Working month: January 2024.
- Worst observed day: 2024-01-26.
- `speed < 30` is the current congestion threshold.
- On 2024-01-26, the sensor had sustained congestion blocks such as 06:10–10:45 and 11:20–17:50.

Current target setup:
- `is_congested = speed < 30`.
- `congestion_30min = is_congested.shift(-6)`.
- `congestion_60min = is_congested.shift(-12)`.
- `congestion_90min = is_congested.shift(-18)`.
- `congestion_120min = is_congested.shift(-24)`.

Current feature setup:
- Calendar features: hour, minute, day_of_week, day, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos.
- Lag features up to 60 minutes: 5, 15, 30, 60 minutes for speed, flow, and occupancy.
- Rolling features up to 60 minutes for speed, flow, and occupancy.
- Difference features such as speed_diff_5min, speed_diff_15min, speed_diff_30min, occupancy_diff_5min, occupancy_diff_15min.

We are currently ready to build the persistence baseline model for one sensor / January 2024.
Please continue from the baseline model step and explain each decision conceptually, not just with code.
```

---

# Current next action

Continue with:

```text
Stage 2 — Baseline model
```

Immediate next tasks:

- [ ] Run temporal train/test split.
- [ ] Build persistence baseline.
- [ ] Evaluate baseline for 30, 60, 90, 120 minutes.
- [ ] Evaluate early-warning subset where the current state is not congested.
- [ ] Save baseline results.
