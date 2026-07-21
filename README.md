# TurbineGuard

Predicts Remaining Useful Life (RUL) for aircraft turbofan engines from sensor data —
how many more flight cycles an engine can run before it needs maintenance. Built on
NASA's C-MAPSS turbofan degradation dataset. Deployed as a FastAPI prediction service.

**Why this matters:** airlines currently run engines on fixed maintenance schedules,
regardless of actual condition — leading to either wasted teardowns on healthy engines
or, worse, missed warning signs on failing ones. Condition-based prediction like this
lets maintenance teams intervene exactly when needed, not on a fixed calendar.

## Live API
`POST /predict` — send sensor readings, get back predicted RUL and a risk tier
(Critical / Warning / Moderate / Healthy).

## Results

| Model | Test MAE (cycles) | Test R² |
|---|---|---|
| Linear Regression (baseline) | 16.4 | 0.77 |
| XGBoost (untuned) | 12.4 | 0.82 |
| XGBoost (tuned) | 12.24 | 0.82 |

Tuning targeted a diagnosed overfitting gap (train MAE 8.0 vs test MAE 12.4) using
shallower trees and row/column subsampling — cut the train-test gap roughly in half
(9.96 vs 12.24) at negligible cost to test accuracy.

Validated with grouped 5-fold cross-validation (average MAE 11.74, std dev 1.26) to
confirm results were representative, not a lucky single split.

## Approach

**Feature engineering** — raw sensor readings are noisy; added rolling averages
(5-cycle window, smooths noise) and rolling slopes (5-cycle linear trend, captures
degradation direction) for all 14 informative sensors. 7 sensors and all 3 operating
settings were dropped after confirming near-zero variance — this dataset uses a single
operating condition, so those columns carried no signal.

**Data leakage checks** — the helper column used to derive RUL (`max_cycle`) was
caught and removed before training, since it would let the model shortcut the answer
via subtraction instead of learning from sensors.

**Train/test split** — split by engine (80/20), not by row, so no engine's cycles
appear in both sets — a random row split would let the model implicitly "see" engines
it's meant to be tested on.

**Unsupervised health clustering** — K-Means (k=4, chosen via elbow method) groups
engines into health states using smoothed sensor trends. Validated against true RUL:
cluster average RUL ranged from 114 (healthiest) to 18.7 (most critical), confirming
the clusters reflect real degradation, not noise. Used as an interpretable label for
the API's risk tier rather than for raw accuracy gains (it didn't meaningfully move
the XGBoost MAE, since XGBoost already had access to the same underlying features).

**Explainability (SHAP)** — TreeExplainer identifies `time_cycles`, `sensor_15`,
`sensor_4`, `sensor_11`, and `sensor_9` (rolling averages) as the strongest predictors
of degradation. Per-engine force plots explain individual predictions — e.g. a
flagged critical engine (predicted RUL: 0.08) shows elevated readings across these
same top sensors driving the prediction down from the ~85-cycle average.

## Tech stack
Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Docker

## Dataset
NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001 subset — single operating
condition, single fault mode, 100 training engines).

## Future work
- Extend to FD002–FD004 (multiple operating conditions, multiple fault modes)
- Supabase (PostgreSQL) data layer in place of local files
- Streamlit demo dashboard calling the live API
