"""
simulator.py — Real-time UPI transaction simulator.
Generates realistic transactions with occasional fraud injection.
Feeds through the full prediction pipeline and stores results in SQLite.
"""

import random
import time
import uuid
from datetime import datetime

from src.database_manager import DatabaseManager
from src.fraud_prediction import predict_fraud

# ── Constants ─────────────────────────────────────────────────────
LOCATIONS = ["Mumbai", "Delhi", "Kolkata", "Lucknow", "Bangalore"]
DEVICES = ["Android_A", "Android_B", "iPhone_X", "iPhone_Y"]
MERCHANTS = ["paytm@upi", "phonepe@upi", "flipkart@upi", "gpay@upi", "amazon@upi"]

# Synthetic user pool with preset behavioral patterns
USER_PROFILES_SEED = [
    {"user_id": 1001, "usual_location": "Mumbai",     "usual_device": "Android_A", "avg_spend": 250},
    {"user_id": 1002, "usual_location": "Delhi",      "usual_device": "iPhone_X",  "avg_spend": 500},
    {"user_id": 1003, "usual_location": "Bangalore",  "usual_device": "Android_B", "avg_spend": 150},
    {"user_id": 1004, "usual_location": "Kolkata",    "usual_device": "iPhone_Y",  "avg_spend": 300},
    {"user_id": 1005, "usual_location": "Lucknow",    "usual_device": "Android_A", "avg_spend": 100},
    {"user_id": 1006, "usual_location": "Mumbai",     "usual_device": "iPhone_X",  "avg_spend": 800},
    {"user_id": 1007, "usual_location": "Delhi",      "usual_device": "Android_B", "avg_spend": 450},
    {"user_id": 1008, "usual_location": "Bangalore",  "usual_device": "Android_A", "avg_spend": 200},
    {"user_id": 1009, "usual_location": "Kolkata",    "usual_device": "iPhone_Y",  "avg_spend": 600},
    {"user_id": 1010, "usual_location": "Lucknow",    "usual_device": "iPhone_X",  "avg_spend": 350},
    {"user_id": 1011, "usual_location": "Mumbai",     "usual_device": "Android_B", "avg_spend": 175},
    {"user_id": 1012, "usual_location": "Delhi",      "usual_device": "Android_A", "avg_spend": 900},
    {"user_id": 1013, "usual_location": "Bangalore",  "usual_device": "iPhone_X",  "avg_spend": 275},
    {"user_id": 1014, "usual_location": "Kolkata",    "usual_device": "iPhone_Y",  "avg_spend": 125},
    {"user_id": 1015, "usual_location": "Lucknow",    "usual_device": "Android_A", "avg_spend": 400},
    {"user_id": 1016, "usual_location": "Mumbai",     "usual_device": "Android_B", "avg_spend": 550},
    {"user_id": 1017, "usual_location": "Delhi",      "usual_device": "iPhone_X",  "avg_spend": 700},
    {"user_id": 1018, "usual_location": "Bangalore",  "usual_device": "iPhone_Y",  "avg_spend": 325},
    {"user_id": 1019, "usual_location": "Kolkata",    "usual_device": "Android_A", "avg_spend": 180},
    {"user_id": 1020, "usual_location": "Lucknow",    "usual_device": "Android_B", "avg_spend": 450},
]


def generate_normal_transaction(user_seed: dict) -> dict:
    """Generate a realistic normal transaction for a user."""
    avg = user_seed["avg_spend"]
    amount = round(random.gauss(avg, avg * 0.3), 2)
    amount = max(5.0, amount)  # minimum ₹5

    hour = random.choices(
        range(24),
        weights=[1, 1, 1, 1, 1, 1, 2, 5, 8, 10, 12, 12, 14, 12, 10, 8, 8, 10, 12, 10, 8, 5, 3, 2],
        k=1
    )[0]

    return {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "user_id": user_seed["user_id"],
        "amount": amount,
        "hour": hour,
        "device_id": user_seed["usual_device"],
        "location": user_seed["usual_location"],
        "merchant_id": random.choice(MERCHANTS),
        "timestamp": datetime.now().isoformat(),
    }


def inject_fraud_patterns(transaction: dict, user_seed: dict) -> dict:
    """Inject suspicious patterns into a transaction to simulate fraud."""
    fraud_txn = transaction.copy()

    # Always spike the amount
    fraud_txn["amount"] = round(user_seed["avg_spend"] * random.uniform(5, 12), 2)

    # Randomly apply additional fraud signals
    if random.random() < 0.7:
        fraud_txn["hour"] = random.randint(0, 5)  # Night hours

    if random.random() < 0.6:
        other_devices = [d for d in DEVICES if d != user_seed["usual_device"]]
        fraud_txn["device_id"] = random.choice(other_devices)

    if random.random() < 0.5:
        other_locations = [l for l in LOCATIONS if l != user_seed["usual_location"]]
        fraud_txn["location"] = random.choice(other_locations)

    return fraud_txn


def process_transaction(db: DatabaseManager, transaction: dict) -> dict:
    """
    Run a transaction through the full pipeline:
    fetch profile → predict → store → update profile.
    """
    user_id = transaction["user_id"]

    # Fetch user behavioral profile
    user_profile = db.get_user_profile(user_id)
    velocity = db.get_transaction_velocity(user_id, transaction["timestamp"])

    # Predict fraud
    result = predict_fraud(transaction, user_profile, velocity)

    # Merge prediction results into transaction record
    full_record = {
        **transaction,
        "fraud_probability": result["fraud_probability"],
        "risk_level": result["risk_level"],
        "explanation": result["explanation"],
    }

    # Store transaction
    db.insert_transaction(full_record)

    # Update user profile
    db.update_user_profile(
        user_id=user_id,
        amount=transaction["amount"],
        device_id=transaction["device_id"],
        location=transaction["location"],
        timestamp=transaction["timestamp"],
    )

    return full_record


def run_simulator(db: DatabaseManager, num_transactions: int = 100,
                  delay: float = 0.5, fraud_ratio: float = 0.10,
                  callback=None):
    """
    Run the transaction simulator.

    Args:
        db: DatabaseManager instance
        num_transactions: number of transactions to generate (0 = infinite)
        delay: seconds between transactions
        fraud_ratio: fraction of transactions that are fraudulent
        callback: optional function called with each processed transaction
    """
    count = 0
    while num_transactions == 0 or count < num_transactions:
        # Pick a random user
        user_seed = random.choice(USER_PROFILES_SEED)

        # Generate transaction
        txn = generate_normal_transaction(user_seed)

        # Inject fraud patterns randomly
        if random.random() < fraud_ratio:
            txn = inject_fraud_patterns(txn, user_seed)

        # Process through pipeline
        result = process_transaction(db, txn)
        count += 1

        if callback:
            callback(result, count)

        time.sleep(delay)

    return count
