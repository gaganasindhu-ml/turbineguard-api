# ✈️ TurbineGuard — Aircraft Engine Failure Prediction

![Model](https://img.shields.io/badge/Model-XGBoost-brightgreen)
![MAE](https://img.shields.io/badge/Test%20MAE-12.24%20cycles-blue)
![Features](https://img.shields.io/badge/Features-43-orange)
![Task](https://img.shields.io/badge/Task-Regression-purple)
![Python](https://img.shields.io/badge/Python-3.11-yellow)
![Deploy](https://img.shields.io/badge/Deploy-FastAPI%20%2B%20Docker-red)
![Live](https://img.shields.io/badge/Live%20API-Render-46E3B7)
![Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)

> **Predicting Remaining Useful Life (RUL) of aircraft turbofan engines from sensor data — the same category of model GE Aviation, Boeing, and Rolls-Royce run in production for condition-based maintenance.**

**🖥️ Live Demo:** [turbineguard-api-zrsmmrrahsakaxpftr7jri.streamlit.app](https://turbineguard-api-zrsmmrrahsakaxpftr7jri.streamlit.app/) — pick a sample engine, get a live prediction with an explained risk tier

**🔴 Live API:** [turbineguard-api.onrender.com](https://turbineguard-api.onrender.com) · **Interactive docs:** [turbineguard-api.onrender.com/docs](https://turbineguard-api.onrender.com/docs)
> Free-tier hosting — the first request after inactivity may take 30-60s to wake up.

---

## 🚨 The Problem

Airlines maintain engines on **fixed calendar schedules**, regardless of actual condition — pull the engine for inspection every N flight cycles, no matter how it's actually performing.

This causes two expensive failure modes:

- **Pulling a healthy engine too early** — wasted maintenance capacity, grounded aircraft, unnecessary teardown cost
- **Missing a failing engine in time** — in the worst case, an in-flight shutdown: emergency diversion, passenger compensation, regulatory scrutiny, and real safety risk

An unplanned engine failure can ground a plane for $150K+/day. **TurbineGuard** predicts, from live sensor readings, exactly how many flight cycles an engine has left — enabling condition-based maintenance instead of guesswork.

---

## 🎯 What It Does

Given an engine's current sensor readings, TurbineGuard's API returns:

- **Predicted Remaining Useful Life** — in flight cycles
- **A risk tier** — Critical / Warning / Moderate / Healthy
- **SHAP-based explanations** — exactly which sensors are driving the prediction, for any individual engine

No black box. Every prediction traces back to specific, physically interpretable sensor behavior.

---

## 📊 Dataset

**Source:** [NASA C-MAPSS Turbofan Engine Degradation Simulation](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

This project uses the **FD001** sub-dataset — 100 engines, single operating condition,
single fault mode (compressor degradation), tracked from healthy operation through
to failure across 21 onboard sensors.

| File | Contents |
|------|----------|
| `train_FD001.txt` | 100 engines, run to failure — used for training and validation |
| 21 sensor channels | Temperature, pressure, fuel flow, and other onboard readings per cycle |
| 3 operational settings | Altitude, speed, throttle — constant in FD001 (single operating condition) |

**After cleaning and feature engineering:** 20,231 rows (100 engines × ~200 cycles
each), 43 model-ready features.

---

## ⚙️ Feature Engineering

Raw sensor readings are noisy day-to-day — what actually signals wear is the **trend**,
not a single reading. Every feature below is engineered, not raw:

| Feature | Why It Matters |
|---------|----------------|
| `RUL` (target) | Derived per-engine as `max_cycle − time_cycles`; capped at 125 so the model focuses learning on the decline phase, where sensors actually carry signal |
| `sensor_X_rollavg` (14 features) | 5-cycle rolling average per sensor — smooths noise, shows current health level |
| `sensor_X_slope` (14 features) | 5-cycle rolling linear-regression slope per sensor — captures direction and speed of degradation, which a rolling average alone hides |
| `time_cycles` | Engine age in flight cycles — the single strongest predictor per SHAP |
| `health_cluster` | K-Means (k=4) health-state label from sensor trends — human-readable risk tier for the API/dashboard layer |

**7 of 21 sensors and all 3 operating settings were dropped** after confirming
near-zero variance directly (not assumed) — FD001's single operating condition means
these columns never actually change, so they carry no learnable signal here.

---

## 🧠 Model Architecture

```
Raw NASA sensor data (train_FD001.txt, no headers)
        ↓
Define column schema (21 sensors, 3 settings, cycle index)
        ↓
Derive RUL target (max_cycle − time_cycles, capped at 125)
        ↓
Drop zero-variance sensors/settings + max_cycle helper column (leakage check)
        ↓
Feature Engineering
(5-cycle rolling averages + rolling slopes, per engine, per sensor)
        ↓
Drop early-life NaN rows (insufficient rolling-window history)
        ↓
Train/Test Split — by engine (80/20), not by row (prevents leakage across cycles)
        ↓
Grouped 5-Fold Cross-Validation (GroupKFold) — trustworthy baseline score
        ↓
XGBoost Regressor — tuned for train/test overfitting gap
(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
        ↓
K-Means Health Clustering (k=4, elbow method) — validated against true RUL separation
        ↓
SHAP TreeExplainer — global + per-engine explanations
        ↓
Export (joblib) → FastAPI /predict endpoint → Docker
```

---

## 📈 Model Performance

| Model | Test MAE (cycles) | Test R² |
|---|---|---|
| Linear Regression (baseline) | 16.39 | 0.767 |
| XGBoost (untuned) | 12.40 | 0.823 |
| **XGBoost (tuned)** | **12.24** | **0.823** |

**Overfitting diagnosis and fix:** the untuned model showed train MAE of 8.00 vs test
MAE of 12.40 — a real ~55% gap. Tuning (shallower trees, row/column subsampling) cut
this gap roughly in half (train 9.96 / test 12.24) at negligible cost to test accuracy —
a more honest, production-trustworthy model, not just a better leaderboard number.

**Validated with grouped 5-fold cross-validation:** average MAE 11.74, std dev 1.26 —
confirms the single train/test split result wasn't a lucky draw.

**Health clustering validated against ground truth:** average RUL by cluster ranged
from 114.3 (healthiest) down to 18.7 (most critical) — confirming the unsupervised
clusters reflect real degradation, not statistical noise.

---

## 🗂️ Project Structure

```
turbineguard-api/
│
├── 🤖 main.py                     # FastAPI app — /predict endpoint
├── 🖥️ app.py                      # Streamlit demo dashboard
├── 📄 requirements.txt            # Python dependencies
├── 🐳 Dockerfile                  # Container build config (FastAPI service)
│
├── turbineguard_model.pkl         # Trained, tuned XGBoost model
├── turbineguard_scaler.pkl        # StandardScaler fit on training data
├── turbineguard_kmeans.pkl        # K-Means health-cluster model (k=4)
├── feature_columns.pkl            # Locked feature order for inference
│
└── 📄 README.md                   # This file
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/gaganasindhu-ml/turbineguard-api.git
cd turbineguard-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`

### Or, with Docker
```bash
docker build -t turbineguard .
docker run -p 8000:8000 turbineguard
```

---

## 🔍 How to Use the API

**Health check:**
```bash
GET /
→ {"status": "TurbineGuard API is running"}
```

**Predict RUL:**
```bash
POST /predict
Content-Type: application/json

{
  "features": {
    "time_cycles": 180,
    "sensor_2": 642.1,
    "sensor_3": 1598.4,
    ...
  }
}
```

**Response:**
```json
{
  "predicted_RUL": 0.08,
  "risk_tier": "Critical"
}
```

---

## 💡 Key Design Decisions & Why They Matter

**Why derive RUL instead of using a given label?**
NASA provides only cycle counts and sensor readings, not a target. Since every training
engine runs to failure, RUL is derivable as `max_cycle − time_cycles` per engine — but
this means the derivation itself has to be leakage-checked (see below), not treated as
a given, trusted column.

**Why was `max_cycle` removed after being used to build the target?**
`max_cycle` is literally the failure day — keeping it as a feature would let the model
shortcut RUL via simple subtraction instead of learning from sensor behavior, and it's
never actually known in advance in production. Classic data leakage, caught before
training rather than after.

**Why drop rows instead of filling missing values with 0?**
Rolling-window features leave the first 4 cycles of every engine as NaN (insufficient
history). Filling with 0 would tell the model a healthy new engine reads "0" on sensors
that actually read in the thousands — a believable but false signal. These rows also
fall in the low-signal "healthy" phase already excluded by the RUL cap, so dropping
them (under 2% of data) was the more honest choice.

**Why split by engine, not by row?**
Consecutive cycles from the same engine look nearly identical. A random row-level split
risks putting some of an engine's cycles in training and others in testing — inflating
test accuracy dishonestly, since the model would have implicitly "met" that engine
already. Splitting by whole engine (80/20) ensures test engines are genuinely unseen.

**Why tune based on cross-validation, not a single test score?**
A single 80/20 split could be a lucky or unlucky draw. Grouped 5-fold cross-validation
(rotating which engines are held out) gave a stable, trustworthy score to tune against,
rather than risking overfitting to one particular split.

**Why keep the health-cluster feature even though it didn't improve XGBoost's accuracy?**
It's derived from features XGBoost already had direct access to, so it added little new
signal (confirmed: MAE moved from 12.40 → 12.234, statistically negligible). It's kept
anyway because it's independently useful as a human-readable summary — a maintenance
crew reads "Critical" instantly, not 43 raw feature values.

---

## 🌍 Business Impact

At fleet scale, condition-based maintenance driven by a model like this can:

- **Avoid unplanned in-flight shutdowns** by flagging critical engines before failure, not after
- **Reduce unnecessary teardowns** on engines that are actually still healthy, despite being due for scheduled inspection
- **Give maintenance crews a specific, explainable reason** for every flag (via SHAP), not just a number — critical for adoption and trust
- **Prioritize a fleet's maintenance queue** by predicted urgency instead of a flat calendar schedule

---

## 🔮 Roadmap

- [x] Feature engineering, leakage-checked target derivation, grouped validation
- [x] Tuned XGBoost with diagnosed overfitting fix
- [x] Unsupervised health clustering, validated against ground truth
- [x] SHAP global + per-engine explainability
- [x] FastAPI prediction service, containerized with Docker
- [x] Streamlit demo dashboard calling the live API, with pre-loaded sample engines
- [ ] Supabase (PostgreSQL) data layer in place of local files
- [ ] Extend to FD002–FD004 (multiple operating conditions, multiple fault modes)

---

## 👤 About

**Gaganasindhu Hadagali**
Data Scientist / AI-ML Engineer

| | |
|--|--|
| GitHub | [github.com/gaganasindhu-ml](https://github.com/gaganasindhu-ml) |

---

## 📄 License

MIT License — free to use, modify, and build on.

---

## 🙏 Acknowledgements

- **Dataset:** NASA Prognostics Center of Excellence — C-MAPSS Turbofan Engine Degradation Simulation
- **SHAP** by Scott Lundberg — model explainability
- **XGBoost** by Tianqi Chen — gradient boosting framework
- **FastAPI** by Sebastián Ramírez — API framework

---

*Turning raw sensor drift into a maintenance decision, before the engine has to make it for you.*
