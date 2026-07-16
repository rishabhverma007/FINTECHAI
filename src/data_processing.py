"""
data_processing.py — Real-time behavioral feature engineering.
Computes features from raw transaction + user profile to match the trained model's 24 columns.
"""

import pandas as pd
import numpy as np

# Known categorical values from the training data
LOCATIONS = ["Bangalore", "Delhi", "Kolkata", "Lucknow", "Mumbai"]
DEVICES = ["Android_A", "Android_B", "iPhone_X", "iPhone_Y"]
MERCHANTS = ["amazon@upi", "flipkart@upi", "gpay@upi", "paytm@upi", "phonepe@upi"]

# Exact feature column order expected by the model
FEATURE_COLUMNS = [
    "amount", "hour", "user_id",
    "avg_user_amount", "amount_deviation", "is_night",
    "is_new_device", "location_change_flag", "is_new_merchant", "transaction_velocity",
    # One-hot: locations
    "location_Bangalore", "location_Delhi", "location_Kolkata", "location_Lucknow", "location_Mumbai",
    # One-hot: devices
    "device_id_Android_A", "device_id_Android_B", "device_id_iPhone_X", "device_id_iPhone_Y",
    # One-hot: merchants
    "merchant_id_amazon@upi", "merchant_id_flipkart@upi",
    "merchant_id_gpay@upi", "merchant_id_paytm@upi", "merchant_id_phonepe@upi",
]


def compute_behavioral_features(transaction: dict, user_profile: dict,
                                 transaction_velocity: int = 1) -> dict:
    """
    Compute behavioral features for a single transaction.

    Args:
        transaction: dict with keys: user_id, amount, hour, device_id, location, merchant_id
        user_profile: dict with keys: avg_amount, last_device, usual_location, transaction_count
        transaction_velocity: number of recent transactions in time window

    Returns:
        dict of all computed feature values
    """
    amount = float(transaction["amount"])
    hour = int(transaction["hour"])
    user_id = int(transaction["user_id"])
    device_id = str(transaction["device_id"])
    location = str(transaction["location"])
    merchant_id = str(transaction["merchant_id"])

    avg_amount = float(user_profile.get("avg_amount", 0.0))
    last_device = str(user_profile.get("last_device", ""))
    usual_location = str(user_profile.get("usual_location", ""))
    txn_count = int(user_profile.get("transaction_count", 0))

    # ── Behavioral features ───────────────────────────────────────
    amount_deviation = abs(amount - avg_amount) / max(avg_amount, 1.0) if avg_amount > 0 else 0.0
    is_night = 1 if hour >= 0 and hour <= 6 else 0
    is_new_device = 1 if (last_device != "" and device_id != last_device) else 0
    location_change_flag = 1 if (usual_location != "" and location != usual_location) else 0
    is_new_merchant = 0  # Conservative: flag only when specifically injected
    velocity = max(transaction_velocity, 1)

    # ── One-hot encoding ──────────────────────────────────────────
    location_ohe = {f"location_{loc}": (1 if location == loc else 0) for loc in LOCATIONS}
    device_ohe = {f"device_id_{dev}": (1 if device_id == dev else 0) for dev in DEVICES}
    merchant_ohe = {f"merchant_id_{m}": (1 if merchant_id == m else 0) for m in MERCHANTS}

    features = {
        "amount": amount,
        "hour": hour,
        "user_id": user_id,
        "avg_user_amount": avg_amount,
        "amount_deviation": round(amount_deviation, 4),
        "is_night": is_night,
        "is_new_device": is_new_device,
        "location_change_flag": location_change_flag,
        "is_new_merchant": is_new_merchant,
        "transaction_velocity": velocity,
        **location_ohe,
        **device_ohe,
        **merchant_ohe,
    }

    return features


def build_feature_dataframe(features: dict) -> pd.DataFrame:
    """
    Convert feature dict to a DataFrame aligned to the model's expected columns.
    Missing columns get 0, extra columns are dropped.
    """
    df = pd.DataFrame([features])
    # Ensure exact column order and fill any missing columns with 0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]
    return df


def generate_explanation(features: dict, fraud_probability: float, threshold: float) -> str:
    """
    Generate human-readable explanation for a fraud prediction.
    Returns a multi-line explanation string.
    """
    explanations = []

    if fraud_probability >= threshold:
        explanations.append("🚨 HIGH RISK TRANSACTION DETECTED")
    else:
        explanations.append("✅ Transaction appears normal")

    explanations.append(f"   Fraud Probability: {fraud_probability:.1%}")

    # Check each behavioral flag
    if features.get("amount_deviation", 0) > 1.5:
        dev = features["amount_deviation"]
        explanations.append(f"   ⚠ Unusual transaction amount ({dev:.1f}x deviation from average)")

    if features.get("is_new_device", 0) == 1:
        explanations.append("   ⚠ New device detected (different from usual device)")

    if features.get("is_night", 0) == 1:
        explanations.append(f"   ⚠ Unusual transaction time (hour: {features.get('hour', '?')}, night hours)")

    if features.get("location_change_flag", 0) == 1:
        explanations.append("   ⚠ Location change detected (different from usual location)")

    if features.get("is_new_merchant", 0) == 1:
        explanations.append("   ⚠ New merchant detected (first-time interaction)")

    if features.get("transaction_velocity", 0) > 5:
        explanations.append(f"   ⚠ Rapid sequential transactions (velocity: {features['transaction_velocity']})")

    if len(explanations) == 2:  # Only header + probability, no flags
        explanations.append("   ℹ No specific behavioral anomalies detected")

    return "\n".join(explanations)
