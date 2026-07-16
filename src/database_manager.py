"""
database_manager.py — SQLite backend for the fraud detection system.
Manages transactions table and user behavioral profiles.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
DB_PATH = os.path.join(DB_DIR, "fraud_detection.db")


class DatabaseManager:
    """Manages SQLite database for transactions and user profiles."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    hour INTEGER NOT NULL,
                    device_id TEXT NOT NULL,
                    location TEXT NOT NULL,
                    merchant_id TEXT NOT NULL,
                    fraud_probability REAL,
                    risk_level TEXT,
                    explanation TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    avg_amount REAL DEFAULT 0.0,
                    last_device TEXT DEFAULT '',
                    usual_location TEXT DEFAULT '',
                    transaction_count INTEGER DEFAULT 0,
                    last_transaction_time TEXT DEFAULT ''
                )
            """)

            # Index for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_user
                ON transactions(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
                ON transactions(timestamp DESC)
            """)

    # ── User Profiles ──────────────────────────────────────────────

    def get_user_profile(self, user_id: int) -> dict:
        """Fetch a user's behavioral profile. Returns defaults if new user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            else:
                return {
                    "user_id": user_id,
                    "avg_amount": 0.0,
                    "last_device": "",
                    "usual_location": "",
                    "transaction_count": 0,
                    "last_transaction_time": "",
                }

    def update_user_profile(self, user_id: int, amount: float, device_id: str,
                            location: str, timestamp: str):
        """Update user profile with new transaction data using running average."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            profile = self.get_user_profile(user_id)

            count = profile["transaction_count"] + 1
            new_avg = ((profile["avg_amount"] * profile["transaction_count"]) + amount) / count

            cursor.execute("""
                INSERT INTO user_profiles (user_id, avg_amount, last_device, usual_location,
                                           transaction_count, last_transaction_time)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    avg_amount = ?,
                    last_device = ?,
                    usual_location = ?,
                    transaction_count = ?,
                    last_transaction_time = ?
            """, (user_id, new_avg, device_id, location, count, timestamp,
                  new_avg, device_id, location, count, timestamp))

    # ── Transactions ───────────────────────────────────────────────

    def insert_transaction(self, transaction: dict):
        """Insert a completed transaction record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transactions
                    (transaction_id, user_id, amount, hour, device_id, location,
                     merchant_id, fraud_probability, risk_level, explanation, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction["transaction_id"],
                transaction["user_id"],
                transaction["amount"],
                transaction["hour"],
                transaction["device_id"],
                transaction["location"],
                transaction["merchant_id"],
                transaction.get("fraud_probability"),
                transaction.get("risk_level"),
                transaction.get("explanation", ""),
                transaction["timestamp"],
            ))

    def get_recent_transactions(self, limit: int = 50) -> list:
        """Get the most recent transactions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_fraud_alerts(self, limit: int = 20) -> list:
        """Get recent high-risk transactions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM transactions
                WHERE risk_level = 'HIGH RISK'
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_fraud_stats(self) -> dict:
        """Get aggregate fraud statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as total FROM transactions")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as cnt FROM transactions WHERE risk_level = 'HIGH RISK'")
            high_risk = cursor.fetchone()["cnt"]

            cursor.execute("SELECT AVG(fraud_probability) as avg_prob FROM transactions")
            avg_prob = cursor.fetchone()["avg_prob"] or 0.0

            cursor.execute("""
                SELECT AVG(fraud_probability) as avg_prob
                FROM transactions WHERE risk_level = 'HIGH RISK'
            """)
            avg_fraud_prob = cursor.fetchone()["avg_prob"] or 0.0

            return {
                "total_transactions": total,
                "high_risk_count": high_risk,
                "fraud_rate": (high_risk / total * 100) if total > 0 else 0.0,
                "avg_probability": avg_prob,
                "avg_fraud_probability": avg_fraud_prob,
            }

    def get_transaction_velocity(self, user_id: int, current_time: str, window_hours: int = 1) -> int:
        """Count transactions by a user in the last N hours."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                dt = datetime.fromisoformat(current_time)
                window_start = (dt - timedelta(hours=window_hours)).isoformat()
            except (ValueError, TypeError):
                window_start = ""

            cursor.execute("""
                SELECT COUNT(*) as cnt FROM transactions
                WHERE user_id = ? AND timestamp >= ?
            """, (user_id, window_start))
            return cursor.fetchone()["cnt"]

    def get_hourly_fraud_distribution(self) -> list:
        """Get fraud counts grouped by hour for analytics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT hour, COUNT(*) as total,
                       SUM(CASE WHEN risk_level = 'HIGH RISK' THEN 1 ELSE 0 END) as fraud_count
                FROM transactions
                GROUP BY hour ORDER BY hour
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_user_risk_summary(self) -> list:
        """Get per-user risk summary for analytics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, COUNT(*) as total_txn,
                       AVG(fraud_probability) as avg_risk,
                       SUM(CASE WHEN risk_level = 'HIGH RISK' THEN 1 ELSE 0 END) as fraud_count
                FROM transactions
                GROUP BY user_id ORDER BY avg_risk DESC LIMIT 20
            """)
            return [dict(row) for row in cursor.fetchall()]

    def clear_all_data(self):
        """Clear all tables (for testing/reset)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions")
            cursor.execute("DELETE FROM user_profiles")
