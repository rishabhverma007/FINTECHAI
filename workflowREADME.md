  <p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🛡️ FINTECHAI — AI Behavioral Fraud Detection Platform

> *An AI behavioral fraud detection platform that learns user transaction patterns and flags suspicious digital payments in real time with explainable insights.*

FINTECHAI is a real-time fraud detection system designed for **UPI and mobile wallet ecosystems**. It uses a pre-trained **Logistic Regression** model with behavioral feature engineering to analyze digital payment transactions and detect anomalies — going beyond simple rule-based checks by learning user spending habits, device preferences, location patterns, and transaction timing.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Behavioral AI** | Analyzes spending patterns, device usage, location, timing, and merchant interactions |
| ⚡ **Real-Time Detection** | Processes transactions instantly with fraud probability scoring |
| 📊 **Interactive Dashboard** | Streamlit-based UI with live monitoring, alerts, and analytics |
| 🔍 **Explainable AI** | Human-readable explanations for every fraud alert |
| 🎯 **Transaction Simulator** | Generates realistic UPI transactions with configurable fraud injection |
| 🗄️ **SQLite Backend** | Lightweight persistent storage for transactions and user profiles |
| 📈 **Rich Analytics** | Plotly charts: probability distributions, hourly patterns, per-user risk, location/merchant breakdowns |

---

## 🏗️ Architecture

```
FINTECHAI/
├── models/
│   └── fraud_detection_model.joblib    # Pre-trained model bundle
├── database/
│   └── fraud_detection.db              # SQLite database (auto-created)
├── src/
│   ├── __init__.py
│   ├── database_manager.py             # SQLite backend & user profiles
│   ├── data_processing.py              # Real-time feature engineering
│   ├── fraud_prediction.py             # Prediction engine & explainability
│   ├── simulator.py                    # UPI transaction simulator
│   └── dashboard.py                    # Streamlit dashboard
├── test_system.py                      # Automated test suite
├── requirements.txt
├── fraud_detection_model.joblib        # Original model file
├── master_synthetic_fraud_dataset.csv  # Training dataset (284K transactions)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- pip (Python package manager)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
python -m streamlit run src/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Automated Tests

```bash
# On Windows (may need encoding flag for emoji output)
set PYTHONIOENCODING=utf-8 && python test_system.py

# On Linux/macOS
python test_system.py
```

---

## 📊 Dashboard Overview

The dashboard provides four main views:

### 📋 Live Transactions
Real-time table of all processed transactions with color-coded risk levels. HIGH RISK rows are highlighted in red for instant visibility.

### 🚨 Fraud Alerts
Dedicated panel showing only flagged transactions with detailed behavioral explanations:
- ⚠ Unusual transaction amount (Nx deviation from average)
- ⚠ New device detected
- ⚠ Unusual transaction time (night hours)
- ⚠ Location change detected
- ⚠ Rapid sequential transactions

### 📝 Manual Entry
Submit individual transactions for instant analysis. Fill in user ID, amount, hour, location, device, and merchant to get a real-time fraud probability score with explanation.

### 📈 Analytics
Interactive Plotly charts including:
- Fraud probability distribution histogram
- Hourly transaction & fraud patterns
- Per-user risk scores
- Location-based fraud breakdown
- Merchant fraud distribution (donut chart)

### 🎛️ Sidebar Controls
- **Simulator**: Configure transaction count, delay, and fraud injection ratio
- **Database**: Clear all data for fresh start
- **Model Info**: View model type, feature count, and threshold

---

## 🧠 How It Works

### Prediction Pipeline

```
Raw Transaction
      │
      ▼
┌─────────────────┐
│ Fetch User      │  ← SQLite user_profiles table
│ Behavioral      │
│ Profile         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature         │  Computes 24 features:
│ Engineering     │  • amount_deviation    • is_night
│                 │  • is_new_device       • location_change_flag
│                 │  • is_new_merchant     • transaction_velocity
│                 │  • One-hot: location, device, merchant
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StandardScaler  │  ← Pre-trained scaler
│ Transform       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Logistic        │  ← Pre-trained classifier
│ Regression      │
│ predict_proba() │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk            │  threshold = 0.65
│ Classification  │  ≥ 0.65 → HIGH RISK
│                 │  < 0.65 → LOW RISK
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Explainable AI  │  Human-readable explanation
│ Layer           │  of flagged behavioral anomalies
└────────┬────────┘
         │
         ▼
  Store in SQLite + Update User Profile
```

### Model Bundle

The pre-trained model (`fraud_detection_model.joblib`) contains:

| Component | Type | Details |
|-----------|------|---------|
| `model` | LogisticRegression | Binary fraud classifier |
| `scaler` | StandardScaler | Feature normalization |
| `threshold` | float | **0.65** decision boundary |
| `feature_columns` | list (24) | Includes one-hot encoded categoricals |

### Training Data Profile

- **284,807** transactions
- **~5%** fraud rate
- **5 cities**: Mumbai, Delhi, Kolkata, Lucknow, Bangalore
- **4 devices**: Android_A, Android_B, iPhone_X, iPhone_Y
- **5 UPI merchants**: paytm@upi, phonepe@upi, flipkart@upi, gpay@upi, amazon@upi

---

## 🎯 Transaction Simulator

The simulator generates realistic UPI-style transactions using a pool of **20 synthetic users**, each with preset behavioral patterns (usual location, preferred device, average spending).

### Normal Transactions
- Amount follows Gaussian distribution around user's average
- Daytime-weighted hour selection
- Consistent device and location

### Fraud Injection (~10% by default)
- **Amount spike**: 5–12× above user average
- **Night hours**: Transactions between 12 AM – 5 AM
- **New device**: Different from user's usual device
- **Location change**: Different city
- Combinations of multiple signals

### Usage

```python
from src.database_manager import DatabaseManager
from src.simulator import run_simulator

db = DatabaseManager()
run_simulator(db, num_transactions=100, delay=0.5, fraud_ratio=0.10)
```

---

## 🔌 Module Reference

### `database_manager.py`

```python
from src.database_manager import DatabaseManager

db = DatabaseManager()
db.get_user_profile(user_id=1001)          # Fetch behavioral profile
db.get_recent_transactions(limit=50)       # Latest transactions
db.get_fraud_alerts(limit=20)              # HIGH RISK only
db.get_fraud_stats()                       # Aggregate statistics
db.get_transaction_velocity(user_id, ts)   # Txns in last hour
db.get_hourly_fraud_distribution()         # For analytics charts
db.get_user_risk_summary()                 # Per-user risk scores
```

### `data_processing.py`

```python
from src.data_processing import compute_behavioral_features, build_feature_dataframe

features = compute_behavioral_features(transaction, user_profile, velocity)
df = build_feature_dataframe(features)  # → DataFrame with 24 columns
```

### `fraud_prediction.py`

```python
from src.fraud_prediction import predict_fraud

result = predict_fraud(transaction, user_profile, velocity)
# result = {
#   "fraud_probability": 0.92,
#   "risk_level": "HIGH RISK",
#   "explanation": "🚨 HIGH RISK TRANSACTION DETECTED\n   ⚠ Unusual amount...",
#   "features": { ... }
# }
```

### `simulator.py`

```python
from src.simulator import process_transaction, run_simulator

# Single transaction through full pipeline
result = process_transaction(db, transaction)

# Batch simulation with callback
run_simulator(db, num_transactions=50, delay=0.3, fraud_ratio=0.10,
              callback=lambda result, count: print(f"#{count}: {result['risk_level']}"))
```

---

## 🧪 Test Results

```
TEST 1: Database Operations          ✅ 6/6 checks passed
TEST 2: Feature Engineering           ✅ 4/4 checks, 24-feature alignment verified
TEST 3: Prediction Engine             ✅ Normal: 9.6% LOW RISK, Suspicious: 100% HIGH RISK
TEST 4: Transaction Simulator         ✅ 10 txns generated, stored, stats verified
Dashboard HTTP Check                  ✅ Status 200 at localhost:8501

🎉 ALL TESTS PASSED
```

---

## 🔮 Future Integration

The system is designed for easy integration with:

- **Payment Gateway APIs** — Replace simulator with live transaction streams
- **Banking Systems** — Connect to core banking transaction feeds
- **Mobile Wallet Platforms** — Direct UPI/wallet API integration
- **Alert Systems** — Push notifications, SMS, or email for fraud alerts
- **Model Retraining** — Periodic retraining pipeline with new transaction data

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML model & preprocessing |
| `joblib` | Model serialization |
| `plotly` | Interactive charts |

---

## 📄 License

This project is for educational and demonstration purposes. It showcases how AI-driven behavioral analytics can enhance fraud prevention in digital payment ecosystems.

---

<p align="center">
  <strong>FINTECHAI</strong> — Protecting Digital Payments with Behavioral AI
</p>
