"""
dashboard.py — Streamlit dashboard for the AI fraud detection system.
Provides live transaction monitoring, fraud alerts, manual entry, and analytics.

Run: streamlit run src/dashboard.py
"""

import sys
import os
import time
import uuid
import threading
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database_manager import DatabaseManager
from src.fraud_prediction import predict_fraud, get_model_info
from src.simulator import (
    process_transaction, run_simulator, USER_PROFILES_SEED,
    LOCATIONS, DEVICES, MERCHANTS,
)

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="CognativeSheild-FinTechAi — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header .subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }
    .hero-header .badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-blue .metric-value { color: #00d2ff; }
    .metric-red .metric-value { color: #ff4757; }
    .metric-green .metric-value { color: #2ed573; }
    .metric-amber .metric-value { color: #ffa502; }

    /* Risk Badges */
    .risk-high {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.8px;
        display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ed573, #27ae60);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.8px;
        display: inline-block;
    }

    /* Alert cards */
    .fraud-alert {
        background: linear-gradient(145deg, #2d1117, #3d1520);
        border-left: 4px solid #ff4757;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 12px rgba(255,71,87,0.15);
    }
    .fraud-alert .alert-header {
        color: #ff4757;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .fraud-alert .alert-details {
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }

    /* Sidebar styling */
    .sidebar-info {
        background: rgba(0, 210, 255, 0.08);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Table styling overrides */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize State ──────────────────────────────────────────────
@st.cache_resource
def get_db():
    return DatabaseManager()


@st.cache_resource
def get_model_metadata():
    return get_model_info()


db = get_db()
model_info = get_model_metadata()


# Initialize session state
if "simulator_running" not in st.session_state:
    st.session_state.simulator_running = False
if "sim_count" not in st.session_state:
    st.session_state.sim_count = 0


# ── Hero Header ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🛡️ FINTECHAI</h1>
    <div class="subtitle">AI Behavioral Fraud Detection Platform for Digital Payments</div>
    <div class="badge">⚡ REAL-TIME MONITORING</div>
</div>
""", unsafe_allow_html=True)


# ── Key Metrics ───────────────────────────────────────────────────
stats = db.get_fraud_stats()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card metric-blue">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{stats['total_transactions']:,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card metric-red">
        <div class="metric-label">High Risk Alerts</div>
        <div class="metric-value">{stats['high_risk_count']:,}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card metric-amber">
        <div class="metric-label">Fraud Rate</div>
        <div class="metric-value">{stats['fraud_rate']:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card metric-green">
        <div class="metric-label">Avg Risk Score</div>
        <div class="metric-value">{stats['avg_probability']:.2%}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    st.markdown(f"""
    <div class="sidebar-info">
        <strong>Model:</strong> {model_info['model_type']}<br>
        <strong>Features:</strong> {model_info['n_features']}<br>
        <strong>Threshold:</strong> {model_info['threshold']}<br>
    </div>
    """, unsafe_allow_html=True)

    # ── Simulator Controls ────────────────────────────────────────
    st.markdown("### 🔄 Transaction Simulator")

    sim_count = st.slider("Transactions to generate", 5, 200, 50, step=5)
    sim_delay = st.slider("Delay between transactions (sec)", 0.1, 2.0, 0.3, step=0.1)
    fraud_ratio = st.slider("Fraud injection ratio", 0.05, 0.30, 0.10, step=0.01)

    if st.button("🚀 Run Simulator", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def sim_callback(result, count):
            progress = count / sim_count
            progress_bar.progress(min(progress, 1.0))
            risk_emoji = "🚨" if result["risk_level"] == "HIGH RISK" else "✅"
            status_text.text(f"{risk_emoji} Txn #{count}: ₹{result['amount']:.2f} → {result['risk_level']}")

        with st.spinner("Simulating transactions..."):
            run_simulator(
                db=db,
                num_transactions=sim_count,
                delay=sim_delay,
                fraud_ratio=fraud_ratio,
                callback=sim_callback,
            )

        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed {sim_count} transactions!")
        st.success(f"Generated {sim_count} transactions!")
        st.rerun()

    st.divider()

    # ── Database Controls ─────────────────────────────────────────
    st.markdown("### 🗃️ Database")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        db.clear_all_data()
        st.success("Database cleared!")
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "FINTECHAI uses behavioral AI to analyze transaction patterns "
        "and detect suspicious digital payments in real time. "
        "The system learns user spending habits, device preferences, "
        "and location patterns to identify anomalies."
    )


# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Transactions",
    "🚨 Fraud Alerts",
    "📝 Manual Entry",
    "📈 Analytics",
])


# ── Tab 1: Live Transactions ─────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📊 Recent Transactions</div>', unsafe_allow_html=True)

    recent = db.get_recent_transactions(limit=100)

    if recent:
        df = pd.DataFrame(recent)
        display_cols = ["transaction_id", "user_id", "amount", "hour", "device_id",
                        "location", "merchant_id", "fraud_probability", "risk_level", "timestamp"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols].copy()

        # Format columns
        df_display["amount"] = df_display["amount"].apply(lambda x: f"₹{x:,.2f}")
        df_display["fraud_probability"] = df_display["fraud_probability"].apply(lambda x: f"{x:.1%}")
        df_display["timestamp"] = df_display["timestamp"].apply(
            lambda x: x[:19] if len(str(x)) > 19 else x
        )

        # Color-code risk level
        def highlight_risk(row):
            if row["risk_level"] == "HIGH RISK":
                return ["background-color: rgba(255,71,87,0.15)"] * len(row)
            return [""] * len(row)

        styled_df = df_display.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=500)

        # Quick stats
        high_count = len(df[df["risk_level"] == "HIGH RISK"])
        low_count = len(df[df["risk_level"] == "LOW RISK"])
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"✅ Low Risk: {low_count} transactions")
        with col2:
            st.error(f"🚨 High Risk: {high_count} transactions")
    else:
        st.info("No transactions yet. Use the simulator or submit a manual transaction to get started!")


# ── Tab 2: Fraud Alerts ──────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🚨 Fraud Alerts — High Risk Transactions</div>',
                unsafe_allow_html=True)

    alerts = db.get_fraud_alerts(limit=30)

    if alerts:
        for alert in alerts:
            prob = alert.get("fraud_probability", 0)
            explanation = alert.get("explanation", "No explanation available")
            ts = str(alert.get("timestamp", ""))[:19]

            st.markdown(f"""
            <div class="fraud-alert">
                <div class="alert-header">
                    🚨 {alert['transaction_id']} — Fraud Probability: {prob:.1%}
                </div>
                <div class="alert-details">
                    <strong>User:</strong> {alert['user_id']} &nbsp;|&nbsp;
                    <strong>Amount:</strong> ₹{alert['amount']:,.2f} &nbsp;|&nbsp;
                    <strong>Location:</strong> {alert['location']} &nbsp;|&nbsp;
                    <strong>Device:</strong> {alert['device_id']} &nbsp;|&nbsp;
                    <strong>Merchant:</strong> {alert['merchant_id']} &nbsp;|&nbsp;
                    <strong>Time:</strong> {ts}<br><br>
                    <pre style="color: rgba(255,255,255,0.8); font-family: 'Inter', monospace; font-size: 0.78rem; white-space: pre-wrap;">{explanation}</pre>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-risk transactions detected yet. Run the simulator to generate some!")


# ── Tab 3: Manual Transaction Entry ──────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📝 Submit a Transaction for Analysis</div>',
                unsafe_allow_html=True)

    with st.form("manual_transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            user_id = st.number_input("User ID", min_value=1, max_value=9999, value=1001, step=1)
            amount = st.number_input("Amount (₹)", min_value=1.0, max_value=100000.0,
                                     value=500.0, step=10.0)

        with col2:
            hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
            location = st.selectbox("Location", LOCATIONS)

        with col3:
            device_id = st.selectbox("Device", DEVICES)
            merchant_id = st.selectbox("Merchant", MERCHANTS)

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True,
                                          type="primary")

    if submitted:
        transaction = {
            "transaction_id": f"MANUAL-{uuid.uuid4().hex[:8].upper()}",
            "user_id": user_id,
            "amount": amount,
            "hour": hour,
            "device_id": device_id,
            "location": location,
            "merchant_id": merchant_id,
            "timestamp": datetime.now().isoformat(),
        }

        result = process_transaction(db, transaction)

        # Display result
        st.divider()
        prob = result["fraud_probability"]
        risk = result["risk_level"]

        if risk == "HIGH RISK":
            st.markdown(f"""
            <div class="fraud-alert">
                <div class="alert-header">
                    🚨 HIGH RISK DETECTED — Fraud Probability: {prob:.1%}
                </div>
                <div class="alert-details">
                    <pre style="color: rgba(255,255,255,0.8); font-family: 'Inter', monospace; font-size: 0.85rem; white-space: pre-wrap;">{result['explanation']}</pre>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ **LOW RISK** — Fraud Probability: {prob:.1%}")
            st.code(result["explanation"], language=None)

        # Show feature details in expander
        with st.expander("🔬 View Computed Features"):
            features = result.get("features", {})
            if not features:
                user_profile = db.get_user_profile(user_id)
                velocity = db.get_transaction_velocity(user_id, transaction["timestamp"])
                pred_result = predict_fraud(transaction, user_profile, velocity)
                features = pred_result.get("features", {})

            feature_df = pd.DataFrame([{
                "Feature": k, "Value": v
            } for k, v in features.items()])
            st.dataframe(feature_df, use_container_width=True)


# ── Tab 4: Analytics ──────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📈 Fraud Analytics Dashboard</div>',
                unsafe_allow_html=True)

    recent_all = db.get_recent_transactions(limit=500)

    if recent_all:
        df_all = pd.DataFrame(recent_all)

        # ── Fraud Probability Distribution ────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                df_all, x="fraud_probability", nbins=30,
                color="risk_level",
                color_discrete_map={"HIGH RISK": "#ff4757", "LOW RISK": "#2ed573"},
                title="Fraud Probability Distribution",
                labels={"fraud_probability": "Fraud Probability", "count": "Count"},
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_size=16,
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Hourly Pattern ────────────────────────────────────────
        with col2:
            hourly = db.get_hourly_fraud_distribution()
            if hourly:
                df_hourly = pd.DataFrame(hourly)
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=df_hourly["hour"], y=df_hourly["total"],
                    name="Total", marker_color="rgba(0,210,255,0.5)",
                ))
                fig_hourly.add_trace(go.Bar(
                    x=df_hourly["hour"], y=df_hourly["fraud_count"],
                    name="Fraud", marker_color="#ff4757",
                ))
                fig_hourly.update_layout(
                    title="Transactions by Hour",
                    barmode="overlay",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    title_font_size=16,
                    xaxis_title="Hour of Day",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

        # ── Per-User Risk ─────────────────────────────────────────
        st.markdown("#### 👤 Per-User Risk Summary")
        user_risk = db.get_user_risk_summary()
        if user_risk:
            df_user = pd.DataFrame(user_risk)
            fig_user = px.bar(
                df_user, x="user_id", y="avg_risk",
                color="fraud_count",
                color_continuous_scale=["#2ed573", "#ffa502", "#ff4757"],
                title="Average Risk Score by User",
                labels={"avg_risk": "Avg Risk Score", "user_id": "User ID",
                        "fraud_count": "Fraud Count"},
            )
            fig_user.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_size=16,
            )
            st.plotly_chart(fig_user, use_container_width=True)

        # ── Location-based fraud ──────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            location_fraud = df_all.groupby("location").agg(
                total=("transaction_id", "count"),
                high_risk=("risk_level", lambda x: (x == "HIGH RISK").sum()),
                avg_prob=("fraud_probability", "mean"),
            ).reset_index()

            fig_loc = px.bar(
                location_fraud, x="location", y=["total", "high_risk"],
                barmode="group",
                color_discrete_sequence=["#00d2ff", "#ff4757"],
                title="Fraud by Location",
            )
            fig_loc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_size=16,
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        with col4:
            merchant_fraud = df_all.groupby("merchant_id").agg(
                total=("transaction_id", "count"),
                high_risk=("risk_level", lambda x: (x == "HIGH RISK").sum()),
            ).reset_index()

            fig_merch = px.pie(
                merchant_fraud, names="merchant_id", values="high_risk",
                title="Fraud Alerts by Merchant",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig_merch.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_size=16,
            )
            st.plotly_chart(fig_merch, use_container_width=True)
    else:
        st.info("No data available yet. Run the simulator to generate transaction data for analytics!")


# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; padding:1rem;'>"
    "FINTECHAI — AI Behavioral Fraud Detection Platform • Built with Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True,
)
