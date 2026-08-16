# Traffic Congestion Forecasting Project — Portfolio Roadmap

_Last updated: 2026-08-15_

## Project idea

Build a machine learning early-warning system that predicts whether congestion will form on a road segment / bridge-like traffic sensor in the next **30, 60, 90, and 120 minutes**.

Final simulated user: traffic operations staff.

> Given current traffic conditions, recent traffic history, calendar context, incidents, weather, and possibly nearby sensors, estimate the probability of congestion forming soon so operators can take preventive actions.

---

# Current status snapshot

We are currently finishing the **one sensor / one month MVP**.

Completed so far:

- [x] Understood the XTraffic `.npy` structure.
- [x] Selected one useful sensor.
- [x] Validated that the sensor shows realistic congestion behavior.
- [x] Defined the first congestion label: `speed < 30`.
- [x] Created future targets for 30, 60, 90, and 120 minutes.
- [x] Created time, lag, rolling, and difference features.
- [x] Built a persistence baseline.
- [x] Trained Logistic Regression models.
- [x] Evaluated early-warning performance.
- [x] Tuned an operational alert threshold.
- [x] Trained XGBoost models.
- [x] Confirmed that XGBoost performed much better than Logistic Regression.

Current working model:

```text
Main model: XGBoost
Operational alert threshold: 0.30
Current scope: selected sensor, January 2024
Next scope: selected sensor, all months of 2024
```

Immediate next work before scaling:

- [ ] Run XGBoost leakage sanity checks.
- [ ] Save exact XGBoost metric tables.
- [ ] Generate final XGBoost probability/alert plots.
- [ ] Generate XGBoost feature importance.
- [ ] Then start Stage 4: scale from January 2024 to all months of 2024.

---

# Dataset context

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

Meaning:

```text
traffic[0, 0, :] = first sensor, first timestamp, all variables
traffic[0, :, :] = first sensor, all timestamps, all variables
traffic[:, :, 2] = all sensors, all timestamps, speed variable
```

The `.npy` traffic file does **not** appear to include explicit timestamps. For January 2024, timestamps were reconstructed because:

```text
31 days × 24 hours × 12 measurements/hour = 8928 measurements
```

Working assumption:

```text
frequency = 5 minutes
2024_p01 = January 2024
```

Still to validate:

- [ ] Confirm the variable order `[flow, occupancy, speed]` from metadata/documentation.
- [ ] Validate all monthly file shapes to confirm `p01 = January`, `p02 = February`, etc.

---

# Selected MVP sensor

A candidate sensor was selected based on speed variation and congestion behavior.

Current candidate:

```text
sensor_id = 819086
month = January 2024
```

Worst observed day:

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

Still to do:

- [ ] Join selected sensor with `sensor_meta_feature.csv` to get freeway, city, direction, lat/lng.

---

# Congestion definition

Initial congestion label:

```python
df_selected["is_congested"] = (df_selected["speed"] < 30).astype(int)
```

Current working threshold:

```text
speed < 30
```

Reason:

```text
Visually, speed below 30 separated normal traffic from sustained low-speed congestion for the selected sensor.
```

Potential future experiments:

```text
speed < 25
speed < 35
speed < 40
```

---

# Observed congestion blocks on 2024-01-26

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

# Target setup

Current horizons:

```python
horizons = {
    "30min": 6,
    "60min": 12,
    "90min": 18,
    "120min": 24,
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
- [x] Calculate target balance for all horizons.

---

# Feature engineering setup

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

Lag features currently used:

```python
lag_steps = {
    "5min": 1,
    "15min": 3,
    "30min": 6,
    "60min": 12,
    "90min": 18,
}
```

Reason:

```text
We initially used lags up to 60 minutes. We later added 90-minute lags because the project includes a 90-minute forecast horizon and the user wanted to test longer historical context.
```

Lag checklist:

- [x] Create speed lags up to 90 minutes.
- [x] Create flow lags up to 90 minutes.
- [x] Create occupancy lags up to 90 minutes.
- [x] Create speed differences: 5, 15, 30 minutes.
- [x] Create occupancy differences: 5, 15 minutes.

Rolling features currently used:

```python
rolling_windows = {
    "15min": 3,
    "30min": 6,
    "60min": 12,
    "90min": 18,
}
```

Rolling checklist:

- [x] Create rolling mean for speed.
- [x] Create rolling std for speed.
- [x] Create rolling mean for flow.
- [x] Create rolling mean for occupancy.
- [x] Use `.shift(1)` before rolling calculations to avoid leakage.
- [x] Drop rows with NaN caused by lags/rolling windows.
- [x] Recreate `df_features` after adding 90-minute features.
- [x] Recreate temporal train/test split after rebuilding `df_features`.
- [x] Inspect missing values after feature engineering.

Important note:

```text
Do not use fillna(0) for lag/rolling NaNs. Those NaNs mean there is not enough historical context yet. Remove those rows instead.
```

---

# Stage 1 — Understand one sensor and one month

Goal: understand the dataset structure and validate that one selected sensor/month is usable.

Status: **Mostly complete**

## Dataset structure

- [x] Download XTraffic / TraffiDent dataset.
- [x] Inspect root files and folders.
- [x] Identify yearly traffic folders: `year_2022`, `year_2023`, `year_2024`.
- [x] Identify incident files: `incidents_y2022.csv`, `incidents_y2023.csv`, `incidents_y2024.csv`.
- [x] Identify sensor metadata file: `sensor_meta_feature.csv`.
- [x] Identify network files: `node_order.npy`, `adj_matrix.npy`, `dis_matrix.npy`.

## Understand node order

- [x] Load `node_order.npy`.
- [x] Confirm shape: `(16972,)`.
- [x] Understand that `node_order[index]` maps matrix position to real `sensor_id`.
- [x] Understand that `traffic[0]` corresponds to sensor `node_order[0]`.

## Understand traffic matrix

- [x] Load one monthly traffic file: `2024_p01.npy`.
- [x] Confirm shape: `(16972, 8928, 3)`.
- [x] Interpret dimensions as `sensor × time × variable`.
- [x] Understand that variable order is assumed to be `[flow, occupancy, speed]`.
- [ ] Validate variable order from official metadata or documentation.
- [x] Reconstruct timestamp for January 2024 using 5-minute frequency.
- [ ] Validate all monthly file shapes to confirm `p01 = January`, `p02 = February`, etc.

## Select a useful sensor

- [x] Create summary statistics across sensors.
- [x] Calculate `speed_mean`, `speed_std`, `speed_min`, `slow_rate_30`, `zero_speed_rate`.
- [x] Filter for sensors with meaningful variation and low missing/zero issues.
- [x] Select candidate sensor: `sensor_id = 819086`.
- [ ] Join selected sensor with `sensor_meta_feature.csv` to get freeway, city, direction, lat/lng.

## Visual EDA

- [x] Create `df_selected` for selected sensor and January 2024.
- [x] Plot speed for worst day: `2024-01-26`.
- [x] Plot occupancy for worst day.
- [x] Plot flow for worst day.
- [x] Confirm speed drops by long blocks.
- [x] Confirm occupancy rises when speed drops.
- [x] Conclude this looks like real congestion, not random data noise.

## Define congestion

- [x] Define initial congestion label: `is_congested = speed < 30`.
- [x] Calculate daily congestion rate.
- [x] Identify sustained congestion blocks.
- [ ] Later test alternative thresholds: 25, 35, 40.

---

# Stage 2 — Future targets and baseline model for one sensor / one month

Goal: transform the problem into multi-horizon classification and build a first comparison point.

Status: **Complete for MVP**

## Future targets

- [x] Create `congestion_30min`.
- [x] Create `congestion_60min`.
- [x] Create `congestion_90min`.
- [x] Create `congestion_120min`.
- [x] Validate target alignment.

## Feature matrix

- [x] Create calendar features.
- [x] Create lag features.
- [x] Create rolling features.
- [x] Create difference features.
- [x] Drop NaNs from lag/rolling and future targets.
- [x] Ensure `timestamp`, `date`, `sensor_id`, and `sensor_index` are not used directly as model features.
- [x] Ensure all model features are numeric.

## Baseline model

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

- [x] Sort data by timestamp.
- [x] Create temporal train/test split.
- [x] Use first 70% of time for training and last 30% for testing.
- [x] Evaluate persistence baseline for `congestion_30min`.
- [x] Evaluate persistence baseline for `congestion_60min`.
- [x] Evaluate persistence baseline for `congestion_90min`.
- [x] Evaluate persistence baseline for `congestion_120min`.
- [x] Calculate accuracy, precision, recall, and F1.
- [x] Evaluate early-warning subset where `is_congested == 0` now.
- [ ] Save baseline results as CSV.

Key insight:

```text
The persistence baseline performs poorly for true early-warning detection because when the current state is uncongested, it always predicts no future congestion. This gives recall = 0 in the early-warning subset.
```

---

# Stage 3 — ML models for one sensor / one month

Goal: check whether real ML models improve over the persistence baseline.

Status: **In progress, main modeling complete**

Models:

- [x] Logistic Regression for each horizon.
- [ ] Random Forest for each horizon. Optional; not needed before scaling.
- [x] XGBoost for each horizon.

Evaluation:

- [x] Use same temporal split as baseline.
- [x] Compare against persistence baseline.
- [x] Evaluate all horizons separately.
- [x] Focus on recall and F1, not only accuracy.
- [x] Pay special attention to false negatives.
- [x] Evaluate early-warning cases separately.
- [x] Tune operational probability threshold.
- [ ] Save all metric tables as CSV.
- [ ] Save key plots to `reports/figures/`.

## Logistic Regression results

### Full test-set comparison: baseline vs Logistic Regression

| Target | Accuracy Baseline | Accuracy LogReg | Precision Baseline | Precision LogReg | Recall Baseline | Recall LogReg | F1 Baseline | F1 LogReg | ROC-AUC LogReg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| congestion_30min | 0.900600 | 0.917104 | 0.882716 | 0.881271 | 0.883495 | 0.930274 | 0.883105 | 0.905109 | 0.971201 |
| congestion_60min | 0.858590 | 0.903976 | 0.833333 | 0.868797 | 0.834069 | 0.911739 | 0.833701 | 0.889750 | 0.957759 |
| congestion_90min | 0.821080 | 0.904726 | 0.788360 | 0.871925 | 0.790451 | 0.908930 | 0.789404 | 0.890043 | 0.950441 |
| congestion_120min | 0.782071 | 0.897599 | 0.742504 | 0.874346 | 0.744474 | 0.885942 | 0.743488 | 0.880105 | 0.948122 |

Interpretation:

```text
Logistic Regression improves over the persistence baseline for all horizons.
The improvement is especially clear for 90-minute and 120-minute horizons.
```

### Logistic Regression early-warning results

Early-warning subset definition:

```python
early_test_df = test_df[test_df["is_congested"] == 0]
```

This means:

```text
Current traffic is not congested.
Question: can the model detect congestion before it forms?
```

| Target | Accuracy | Precision | Recall | F1 | ROC-AUC | Positive Rate True | Positive Rate Pred | Rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| congestion_30min | 0.941253 | 0.664062 | 0.643939 | 0.653846 | 0.963301 | 0.086162 | 0.083551 | 1532 |
| congestion_60min | 0.943864 | 0.777174 | 0.760638 | 0.768817 | 0.976527 | 0.122715 | 0.120104 | 1532 |
| congestion_90min | 0.954308 | 0.881279 | 0.814346 | 0.846491 | 0.977678 | 0.154700 | 0.142950 | 1532 |
| congestion_120min | 0.954308 | 0.958159 | 0.792388 | 0.867424 | 0.983503 | 0.188642 | 0.156005 | 1532 |

Interpretation:

```text
Logistic Regression successfully detects future congestion in the early-warning subset, while the baseline has recall = 0 there.
```

## Operational threshold tuning

Default threshold:

```text
0.50
```

Selected operational threshold:

```text
0.30
```

Reason:

```text
For traffic operations, false negatives are costly. A lower threshold increases sensitivity and raises alerts earlier, accepting some false positives.
```

Checklist:

- [x] Test thresholds from 0.10 to 0.90.
- [x] Plot precision, recall, and F1 by threshold.
- [x] Select threshold 0.30 as operational threshold.
- [ ] Save tuned threshold metrics.

## XGBoost results

Status:

```text
XGBoost was trained and reported to be much better than Logistic Regression.
```

Important: exact XGBoost metric values still need to be pasted/saved into this roadmap.

Checklist:

- [x] Install XGBoost.
- [x] Train one XGBoost classifier per horizon.
- [x] Compare XGBoost to Logistic Regression.
- [x] Confirm XGBoost is the current main model.
- [ ] Paste exact XGBoost metric table here.
- [ ] Evaluate XGBoost early-warning subset with default threshold.
- [ ] Evaluate XGBoost early-warning subset with threshold 0.30.
- [ ] Run leakage sanity checks.
- [ ] Generate XGBoost probability/alert plot for one day.
- [ ] Generate feature importance for `congestion_60min`.
- [ ] Save XGBoost results as CSV.

Leakage sanity checks to run before scaling:

```python
[c for c in feature_cols if "congestion" in c or "target" in c or "future" in c]
```

Expected:

```python
[]
```

Also run:

```python
[c for c in feature_cols if c in ["timestamp", "date", "sensor_id", "sensor_index"]]
```

Expected:

```python
[]
```

---

# Stage 4 — Scale from one month to all months of 2024

Goal: turn the current notebook logic into reusable functions and test whether the model generalizes beyond January.

Status: **Next major stage**

Why this matters:

```text
A model that performs well on January 2024 might only have learned January-specific patterns.
Using all months of 2024 gives a stronger evaluation and prepares the project for multi-year training.
```

Required functions:

- [ ] `load_traffic_month(year, month)`
- [ ] `load_sensor_month(year, month, sensor_index)`
- [ ] `build_timestamp_index(year, month, n_rows)`
- [ ] `create_congestion_label(df, threshold=30)`
- [ ] `create_future_targets(df, horizons)`
- [ ] `create_time_features(df)`
- [ ] `create_lag_features(df, lag_steps)`
- [ ] `create_rolling_features(df, rolling_windows)`
- [ ] `create_difference_features(df)`
- [ ] `build_feature_matrix(df)`
- [ ] `temporal_train_test_split(df)`
- [ ] `evaluate_model(model, X, y)`

Then:

- [ ] Validate monthly file shapes for 2024.
- [ ] Load all months of 2024 for selected sensor.
- [ ] Concatenate all months into one DataFrame.
- [ ] Rebuild targets across month boundaries carefully.
- [ ] Rebuild features across month boundaries carefully.
- [ ] Train/test split by date.
- [ ] Re-evaluate persistence baseline.
- [ ] Re-evaluate Logistic Regression.
- [ ] Re-evaluate XGBoost.
- [ ] Compare performance against January-only model.
- [ ] Decide whether January results were stable or over-optimistic.

Important implementation note:

```text
Targets and lags should be created after concatenating months, not separately inside each month, otherwise boundary rows can lose useful context or create artificial gaps.
```

---

# Stage 5 — Scale to 2022, 2023, and 2024

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
- [ ] Compare performance against one-month and one-year models.

---

# Stage 6 — Add incident reports

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
- [ ] Join selected sensor metadata to incidents by freeway/direction/location.
- [ ] Join incidents to sensor using time proximity.
- [ ] Join incidents to sensor using spatial proximity or freeway/direction.
- [ ] Create incident features.
- [ ] Train model with and without incident features.
- [ ] Measure whether incidents improve early-warning performance.

---

# Stage 7 — Add calendar and holiday features

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

# Stage 8 — Add weather data

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

# Stage 9 — Add nearby sensors / spatial context

Goal: use upstream/downstream traffic conditions to predict congestion earlier.

Network files:

```text
adj_matrix.npy
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

# Stage 10 — Model comparison and final evaluation

Goal: produce a strong portfolio-ready evaluation.

Model groups:

- [x] Persistence baseline.
- [x] Logistic Regression.
- [ ] Random Forest. Optional.
- [x] XGBoost.
- [ ] XGBoost + incidents.
- [ ] XGBoost + weather.
- [ ] XGBoost + spatial features.
- [ ] Optional: LSTM.
- [ ] Optional: graph-based model.

Comparison table template:

```text
Model                         30min   60min   90min   120min
Persistence baseline           ...     ...     ...      ...
Logistic Regression            ...     ...     ...      ...
XGBoost                        ...     ...     ...      ...
XGBoost + incidents            ...     ...     ...      ...
XGBoost + weather              ...     ...     ...      ...
XGBoost + spatial features     ...     ...     ...      ...
```

Metrics:

- [x] Accuracy.
- [x] Precision.
- [x] Recall.
- [x] F1.
- [x] ROC-AUC for Logistic Regression.
- [ ] ROC-AUC for XGBoost saved to roadmap.
- [ ] PR-AUC if classes are imbalanced.
- [ ] Confusion matrix by horizon.
- [x] Early-warning recall.
- [ ] Operational-threshold metrics.

---

# Stage 11 — Turn notebook into project structure

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

# Stage 12 — Optional MLOps / API layer

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

df_selected = pd.DataFrame(
    selected_data,
    columns=["flow", "occupancy", "speed"]
)

df_selected["timestamp"] = pd.date_range(
    start="2024-01-01",
    periods=len(df_selected),
    freq="5min"
)

df_selected["sensor_index"] = selected_sensor_index
df_selected["sensor_id"] = selected_sensor_id
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
    "120min": 24,
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
    "60min": 12,
    "90min": 18,
}

for lag_name, steps in lag_steps.items():
    df_model[f"speed_lag_{lag_name}"] = df_model["speed"].shift(steps)
    df_model[f"flow_lag_{lag_name}"] = df_model["flow"].shift(steps)
    df_model[f"occupancy_lag_{lag_name}"] = df_model["occupancy"].shift(steps)
```

## Create XGBoost alert outputs

```python
OPERATIONAL_THRESHOLD = 0.30

df_alerts_xgb = test_df.copy()
df_alerts_xgb["timestamp"] = pd.to_datetime(df_alerts_xgb["timestamp"])

for target in target_cols:
    horizon = target.replace("congestion_", "")
    model = xgb_models[target]

    proba_col = f"xgb_proba_{horizon}"
    alert_col = f"xgb_alert_{horizon}"

    df_alerts_xgb[proba_col] = model.predict_proba(
        df_alerts_xgb[feature_cols]
    )[:, 1]

    df_alerts_xgb[alert_col] = (
        df_alerts_xgb[proba_col] >= OPERATIONAL_THRESHOLD
    ).astype(int)
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
- Lag features: 5, 15, 30, 60, and 90 minutes for speed, flow, and occupancy.
- Rolling features: 15, 30, 60, and 90 minutes for speed, flow, and occupancy.
- Difference features: speed_diff_5min, speed_diff_15min, speed_diff_30min, occupancy_diff_5min, occupancy_diff_15min.

Modeling status:
- Persistence baseline was built.
- Logistic Regression was trained and improved over baseline.
- Logistic Regression early-warning evaluation worked; baseline had recall 0 in early-warning cases.
- Threshold tuning was performed.
- Operational alert threshold selected: 0.30.
- XGBoost was trained and performed much better than Logistic Regression.
- XGBoost is now the current main model.

Immediate next tasks:
1. Run XGBoost leakage sanity checks.
2. Save exact XGBoost metric tables.
3. Generate XGBoost alert probability plots using threshold 0.30.
4. Generate XGBoost feature importance.
5. Then start Stage 4: refactor the one-month notebook into reusable functions and scale from January 2024 to all months of 2024.

Please continue from the XGBoost validation / Stage 4 preparation step and explain each decision conceptually, not just with code.
```

---

# Current next action

Continue with:

```text
Finalize Stage 3 XGBoost validation, then start Stage 4 — scale from January 2024 to all months of 2024.
```

Immediate next tasks:

- [ ] Run leakage sanity checks for XGBoost.
- [ ] Save `xgb_results_df`, `xgb_early_results_df`, and `xgb_tuned_early_results_df`.
- [ ] Plot XGBoost predicted probabilities for a selected test day.
- [ ] Plot XGBoost feature importance.
- [ ] Create reusable data-loading and feature-engineering functions.
- [ ] Load all 2024 months for selected sensor.
