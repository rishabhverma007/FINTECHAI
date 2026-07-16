"""
fraud_prediction.py — Prediction engine with explainable AI.
Loads the pre-trained model bundle and provides fraud probability + risk classification.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.data_processing import (
    compute_behavioral_features,
    build_feature_dataframe,
    generate_explanation,
    FEATURE_COLUMNS,
)

# ── Load model bundle once at module level ────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_model.joblib")

_bundle = None


def _load_model():
    """Lazy-load the model bundle."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_model_info() -> dict:
    """Return model metadata."""
    bundle = _load_model()
    return {
        "model_type": type(bundle["model"]).__name__,
        "scaler_type": type(bundle["scaler"]).__name__,
        "threshold": bundle["threshold"],
        "n_features": len(bundle["feature_columns"]),
        "feature_columns": bundle["feature_columns"],
    }


def predict_fraud(transaction: dict, user_profile: dict,
                  transaction_velocity: int = 1) -> dict:
    """
    Full prediction pipeline for a single transaction.

    Args:
        transaction: raw transaction dict (user_id, amount, hour, device_id, location, merchant_id)
        user_profile: user behavioral profile from DB
        transaction_velocity: recent transaction count for this user

    Returns:
        dict with fraud_probability, risk_level, explanation, and computed features
    """
    bundle = _load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]

    # Step 1: Compute behavioral features
    features = compute_behavioral_features(transaction, user_profile, transaction_velocity)

    # Step 2: Build aligned DataFrame
    features_df = build_feature_dataframe(features)

    # Step 3: Scale features (use .values to avoid feature name warning)
    features_scaled = scaler.transform(features_df.values)

    # Step 4: Predict probability
    proba = model.predict_proba(features_scaled)[0]
    fraud_probability = float(proba[1])  # Probability of class 1 (fraud)

    # Step 5: Classify risk
    risk_level = "HIGH RISK" if fraud_probability >= threshold else "LOW RISK"

    # Step 6: Generate explanation
    explanation = generate_explanation(features, fraud_probability, threshold)

    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_level": risk_level,
        "explanation": explanation,
        "features": features,
    }


def batch_predict(transactions: list, user_profiles: dict,
                  velocities: dict = None) -> list:
    """
    Predict fraud for multiple transactions.

    Args:
        transactions: list of transaction dicts
        user_profiles: dict mapping user_id -> profile dict
        velocities: dict mapping user_id -> velocity count

    Returns:
        list of prediction result dicts
    """
    results = []
    for txn in transactions:
        uid = txn["user_id"]
        profile = user_profiles.get(uid, {
            "avg_amount": 0.0, "last_device": "", "usual_location": "",
            "transaction_count": 0
        })
        velocity = (velocities or {}).get(uid, 1)
        result = predict_fraud(txn, profile, velocity)
        results.append(result)
    return results
