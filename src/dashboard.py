"""
dashboard.py — Streamlit dashboard for the AI fraud detection system.
Provides live transaction monitoring, fraud alerts, manual entry, analytics, and transaction drill-down.

Run: streamlit run src/dashboard.py
"""

import sys
import os
import time
import uuid
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
    process_transaction, BackgroundSimulator, USER_PROFILES_SEED,
    LOCATIONS, DEVICES, MERCHANTS,
)

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ COGNITIVE SHIELD — Real-time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Rich Premium Aesthetics ────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0d0e15;
        color: #e2e8f0;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #1e1b4b, #311042, #0f172a);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(99,102,241,0.3);
    }
    .hero-header .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-header .badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.4rem 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-weight: 600;
    }
    .metric-blue .metric-value { color: #6366f1; text-shadow: 0 0 10px rgba(99,102,241,0.2); }
    .metric-red .metric-value { color: #f43f5e; text-shadow: 0 0 10px rgba(244,63,94,0.2); }
    .metric-green .metric-value { color: #10b981; text-shadow: 0 0 10px rgba(16,185,129,0.2); }
    .metric-amber .metric-value { color: #f59e0b; text-shadow: 0 0 10px rgba(245,158,11,0.2); }

    /* Custom Rules Violations List */
    .violation-tag {
        background-color: rgba(244, 63, 94, 0.15);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Risk Badges */
    .risk-high {
        background: linear-gradient(135deg, #f43f5e, #be123c);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.8px;
        box-shadow: 0 4px 10px rgba(244,63,94,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981, #047857);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.8px;
        box-shadow: 0 4px 10px rgba(16,185,129,0.3);
    }

    /* Alert cards */
    .fraud-alert {
        background: linear-gradient(145deg, #2d141e, #1e0f15);
        border-left: 5px solid #f43f5e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(244, 63, 94, 0.1);
        border-bottom: 1px solid rgba(244, 63, 94, 0.1);
        border-right: 1px solid rgba(244, 63, 94, 0.1);
        box-shadow: 0 10px 30px rgba(244, 63, 94, 0.08);
        transition: all 0.2s ease;
    }
    .fraud-alert:hover {
        transform: scale(1.005);
        box-shadow: 0 12px 35px rgba(244, 63, 94, 0.12);
    }
    .fraud-alert .alert-header {
        color: #f43f5e;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .fraud-alert .alert-details {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 1.8rem 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: -0.3px;
    }

    /* Sidebar info */
    .sidebar-info {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.85rem;
    }

    /* Auto-refresh Dot Animation */
    .refresh-indicator {
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        display: inline-block;
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 0.5; }
        50% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 12px #10b981; }
        100% { transform: scale(0.9); opacity: 0.5; }
    }

    /* Details Grid styling */
    .drilldown-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .drilldown-cell {
        background: rgba(30, 41, 59, 0.25);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
    }
    .drilldown-cell .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .drilldown-cell .value {
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        margin-top: 0.2rem;
    }

    pre, code {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Streamlit overrides */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Initialize State & DB ─────────────────────────────────────────
@st.cache_resource
def get_db():
    return DatabaseManager()


@st.cache_resource
def get_model_metadata():
    return get_model_info()


db = get_db()
model_info = get_model_metadata()

# Initialize session state variables
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# ── Reusable Explainable AI Visualizer ─────────────────────────────
def render_xai_chart(contributions: dict):
    """Render a premium horizontal Plotly bar chart of feature contributions."""
    # Sort contributions by absolute value
    sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_contrib = sorted_contrib[:10]  # Show top 10 contributing features

    name_mapping = {
        "amount": "Transaction Amount",
        "hour": "Hour of Day",
        "user_id": "User ID",
        "avg_user_amount": "User Average Amount",
        "amount_deviation": "Amount Deviation",
        "is_night": "Night Transaction",
        "is_new_device": "New Device Flag",
        "location_change_flag": "Location Change Flag",
        "is_new_merchant": "New Merchant Flag",
        "transaction_velocity": "Transaction Velocity",
    }

    formatted_contribs = []
    for k, v in top_contrib:
        name = k
        if k.startswith("location_"):
            name = f"Location: {k.replace('location_', '')}"
        elif k.startswith("device_id_"):
            name = f"Device: {k.replace('device_id_', '')}"
        elif k.startswith("merchant_id_"):
            name = f"Merchant: {k.replace('merchant_id_', '')}"
        else:
            name = name_mapping.get(k, k)

        formatted_contribs.append({"Feature": name, "Contribution": v})

    df_contrib = pd.DataFrame(formatted_contribs)
    # Reverse to make highest contributors appear at the top
    df_contrib = df_contrib.iloc[::-1]

    # Color code: Positive contributions push log-odds higher (red), negative lower (green)
    colors = ['#f43f5e' if val >= 0 else '#10b981' for val in df_contrib["Contribution"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_contrib["Feature"],
        x=df_contrib["Contribution"],
        orientation='h',
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Influence Score: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Explainable AI — Feature Influence on Risk Score</b>",
        xaxis_title="Influence Score (Red increases risk, Green decreases risk)",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_font_size=15,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.25)",
            zerolinewidth=2
        ),
        yaxis=dict(showgrid=False),
        margin=dict(l=150, r=20, t=50, b=50),
        height=380,
    )
    return fig


# ── Hero Header ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🛡️ COGNITIVE SHIELD</h1>
    <div class="subtitle">Next-Gen Hybrid ML & Rule Fraud Intelligence System for Digital Payments</div>
    <div class="badge">🧠 EXPLAINABLE AI INTEGRATED</div>
</div>
""", unsafe_allow_html=True)


# ── Global Key Metrics Panel ──────────────────────────────────────
stats = db.get_fraud_stats()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card metric-blue">
        <div class="metric-label">Total Volume</div>
        <div class="metric-value">{stats['total_transactions']:,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card metric-red">
        <div class="metric-label">Fraud Alerts Flagged</div>
        <div class="metric-value">{stats['high_risk_count']:,}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card metric-amber">
        <div class="metric-label">System Fraud Rate</div>
        <div class="metric-value">{stats['fraud_rate']:.2f}%</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card metric-green">
        <div class="metric-label">Avg Transaction Risk</div>
        <div class="metric-value">{stats['avg_probability']:.2%}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar Control Panel ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ System Control Console")

    st.markdown(f"""
    <div class="sidebar-info">
        <strong>Model Bundle:</strong> {model_info['model_type']}<br>
        <strong>Feature Dimensions:</strong> {model_info['n_features']}<br>
        <strong>Risk Cutoff Threshold:</strong> {model_info['threshold']:.2f}<br>
    </div>
    """, unsafe_allow_html=True)

    # ── Background Thread Simulator Controls ──────────────────────
    st.markdown("### 🔄 Continuous Live Feed")
    is_sim_active = BackgroundSimulator.is_running()

    sim_delay = st.slider("Simulation Delay (sec)", 0.2, 5.0, 1.0, step=0.1)
    fraud_ratio = st.slider("Fraud Injection Ratio", 0.05, 0.40, 0.15, step=0.01)

    if is_sim_active:
        if st.button("🛑 STOP SIMULATOR FEED", use_container_width=True, type="secondary"):
            BackgroundSimulator.stop()
            st.session_state.auto_refresh = False
            st.success("Background simulator stopped.")
            st.rerun()
        st.markdown("<div style='color:#f43f5e; font-weight:800; text-align:center; font-size:0.85rem; margin-top:8px;'>● LIVE BACKGROUND SIMULATION ACTIVE</div>", unsafe_allow_html=True)
    else:
        if st.button("🚀 START SIMULATOR FEED", use_container_width=True, type="primary"):
            BackgroundSimulator.start(db, delay=sim_delay, fraud_ratio=fraud_ratio)
            st.session_state.auto_refresh = True  # Automatically enable refresh for real-time visualization
            st.success("Background simulator started!")
            st.rerun()
        st.markdown("<div style='color:#64748b; text-align:center; font-size:0.85rem; margin-top:8px;'>Simulator is Idle</div>", unsafe_allow_html=True)

    st.divider()

    # ── Auto-Refresh Controls ─────────────────────────────────────
    st.markdown("### 📡 Live Screen Refresh")
    st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh (2s)", value=st.session_state.auto_refresh)

    if st.session_state.auto_refresh:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: flex-start; gap: 8px; margin-top:4px;">
            <span class="refresh-indicator"></span>
            <span style="color: #10b981; font-size: 0.85rem; font-weight: 600;">Polled Live Database Feed</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Database Actions ──────────────────────────────────────────
    st.markdown("### 🗃️ DB Maintenance")
    if st.button("🗑️ Clear Transactions & Profiles", use_container_width=True):
        if is_sim_active:
            BackgroundSimulator.stop()
        db.clear_all_data()
        st.success("SQLite tables successfully cleared!")
        st.session_state.auto_refresh = False
        st.rerun()


# ── Tabs Configuration ────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Transactions",
    "🚨 Fraud Alerts Feed",
    "🔍 Transaction Drill-Down",
    "📝 Manual Entry Panel",
    "📈 Advanced Analytics",
])


# ── Tab 1: Live Transactions Table ────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📊 Real-Time Transaction Stream</div>', unsafe_allow_html=True)

    recent = db.get_recent_transactions(limit=100)

    if recent:
        df = pd.DataFrame(recent)
        display_cols = ["transaction_id", "user_id", "amount", "hour", "device_id",
                        "location", "merchant_id", "fraud_probability", "risk_level", "timestamp"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols].copy()

        # Format presentation
        df_display["amount"] = df_display["amount"].apply(lambda x: f"₹{x:,.2f}")
        df_display["fraud_probability"] = df_display["fraud_probability"].apply(lambda x: f"{x:.2%}")
        df_display["timestamp"] = df_display["timestamp"].apply(lambda x: x[:19] if len(str(x)) > 19 else x)

        # Highlight high risk entries
        def highlight_rows(row):
            if row["risk_level"] == "HIGH RISK":
                return ["background-color: rgba(244,63,94,0.12)"] * len(row)
            return [""] * len(row)

        styled_df = df_display.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=520)

        c1, c2 = st.columns(2)
        with c1:
            st.info(f"✅ Low Risk Normal Transactions: {len(df[df['risk_level'] == 'LOW RISK'])}")
        with c2:
            st.error(f"🚨 Flagged High Risk Violations: {len(df[df['risk_level'] == 'HIGH RISK'])}")
    else:
        st.info("No transaction data found in SQLite. Activate the background feed simulator to stream payments!")


# ── Tab 2: Fraud Alerts Feed ──────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🚨 Flagged High Risk Behavioral Alerts</div>', unsafe_allow_html=True)

    alerts = db.get_fraud_alerts(limit=25)

    if alerts:
        for alert in alerts:
            prob = alert.get("fraud_probability", 0)
            explanation = alert.get("explanation", "No details")
            ts = str(alert.get("timestamp", ""))[:19]

            # Reconstruct details to fetch rule violations
            user_id = alert["user_id"]
            user_profile = db.get_user_profile(user_id)
            velocity = db.get_transaction_velocity(user_id, alert["timestamp"])
            prediction_details = predict_fraud(alert, user_profile, velocity)
            rule_violations = prediction_details.get("rule_violations", [])

            # Generate formatted rule badges
            badges_html = ""
            for violation in rule_violations:
                badges_html += f'<span class="violation-tag">⚠ {violation["rule_name"]}</span>'

            st.markdown(f"""
            <div class="fraud-alert">
                <div class="alert-header">
                    🚨 Alert: {alert['transaction_id']} &nbsp;|&nbsp; Risk: {prob:.2%}
                </div>
                <div class="alert-details">
                    <strong>User:</strong> ID {alert['user_id']} &nbsp;|&nbsp;
                    <strong>Amount:</strong> ₹{alert['amount']:,.2f} &nbsp;|&nbsp;
                    <strong>Location:</strong> {alert['location']} &nbsp;|&nbsp;
                    <strong>Device:</strong> {alert['device_id']} &nbsp;|&nbsp;
                    <strong>Merchant:</strong> {alert['merchant_id']} &nbsp;|&nbsp;
                    <strong>Time:</strong> {ts}<br>
                    <div style="margin-top: 8px; margin-bottom: 8px;">{badges_html}</div>
                    <pre style="background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; color: #e2e8f0; font-size: 0.8rem; margin: 0; white-space: pre-wrap;">{explanation}</pre>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-risk transactions have been detected yet.")


# ── Tab 3: Transaction Drill-down & In-Depth XAI ──────────────────
with tab3:
    st.markdown('<div class="section-header">🔍 In-Depth Transaction Investigation & Explainable AI (XAI)</div>', unsafe_allow_html=True)

    recent_all = db.get_recent_transactions(limit=150)

    if recent_all:
        txn_ids = [t["transaction_id"] for t in recent_all]
        selected_id = st.selectbox("Select Transaction ID to investigate:", txn_ids)

        # Retrieve selected transaction record
        selected_txn = next(t for t in recent_all if t["transaction_id"] == selected_id)
        user_id = selected_txn["user_id"]

        # Fetch profile and rebuild dynamic explanations
        user_profile = db.get_user_profile(user_id)
        velocity = db.get_transaction_velocity(user_id, selected_txn["timestamp"])
        details = predict_fraud(selected_txn, user_profile, velocity)

        # Layout grid for transaction features
        st.markdown("### 📊 Transaction Properties")
        st.markdown(f"""
        <div class="drilldown-grid">
            <div class="drilldown-cell">
                <div class="label">Transaction ID</div>
                <div class="value" style="font-family: 'JetBrains Mono', monospace; font-size:0.95rem;">{selected_txn['transaction_id']}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Amount</div>
                <div class="value" style="color: #6366f1;">₹{selected_txn['amount']:,.2f}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Risk Probability</div>
                <div class="value" style="color: {'#f43f5e' if details['risk_level'] == 'HIGH RISK' else '#10b981'};">{details['fraud_probability']:.2%}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Status</div>
                <div class="value">
                    <span class="{'risk-high' if details['risk_level'] == 'HIGH RISK' else 'risk-low'}">{details['risk_level']}</span>
                </div>
            </div>
        </div>
        <div class="drilldown-grid">
            <div class="drilldown-cell">
                <div class="label">User Profile ID</div>
                <div class="value">{selected_txn['user_id']}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Device Fingerprint</div>
                <div class="value">{selected_txn['device_id']}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Merchant Address</div>
                <div class="value">{selected_txn['merchant_id']}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Transaction Time</div>
                <div class="value" style="font-size:0.85rem;">{selected_txn['timestamp'][:19]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### 🚨 Behavioral Assessment")
            # Rule violations
            if details["rule_violations"]:
                st.warning("⚠️ **Hard Business Rule Violations Triggered:**")
                for rule in details["rule_violations"]:
                    st.markdown(f"**• {rule['rule_name']}**\n*{rule['description']}*")
            else:
                st.success("✅ **Passed all Hard Business Rule checks.**")

            st.markdown("### 📝 Behavioral Explanation")
            st.text_area("Audit Explanation", value=details["explanation"], height=160, label_visibility="collapsed")

        with col2:
            # Render visual XAI Chart
            fig_xai = render_xai_chart(details["contributions"])
            st.plotly_chart(fig_xai, use_container_width=True)

        # Render user profile context
        st.markdown("### 👤 Historical User Profile Context")
        st.markdown(f"""
        <div class="drilldown-grid">
            <div class="drilldown-cell">
                <div class="label">Average Spend</div>
                <div class="value">₹{user_profile['avg_amount']:,.2f}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Usual device</div>
                <div class="value">{user_profile['last_device'] or 'N/A'}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Usual location</div>
                <div class="value">{user_profile['usual_location'] or 'N/A'}</div>
            </div>
            <div class="drilldown-cell">
                <div class="label">Total Historic Txns</div>
                <div class="value">{user_profile['transaction_count']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("No transaction records available. Generate transactions to examine detailed logs.")


# ── Tab 4: Manual Transaction Form & Prediction ─────────────────
with tab4:
    st.markdown('<div class="section-header">📝 Sandbox Manual Payment Entry</div>', unsafe_allow_html=True)

    with st.form("manual_transaction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            u_id = st.number_input("User Account ID", min_value=1, max_value=9999, value=1001, step=1)
            amt = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=150000.0, value=850.0, step=50.0)

        with c2:
            txn_hour = st.slider("Payment Hour (24h Format)", 0, 23, 12)
            loc = st.selectbox("Location City", LOCATIONS)

        with c3:
            dev_id = st.selectbox("Payment Device Model", DEVICES)
            merch_id = st.selectbox("Payment Gateway Target", MERCHANTS)

        submitted = st.form_submit_button("🔍 Run Fraud Risk Analysis", use_container_width=True, type="primary")

    if submitted:
        transaction = {
            "transaction_id": f"MANUAL-{uuid.uuid4().hex[:8].upper()}",
            "user_id": u_id,
            "amount": amt,
            "hour": txn_hour,
            "device_id": dev_id,
            "location": loc,
            "merchant_id": merch_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Process through simulator's transaction flow to store in database and update profile
        result = process_transaction(db, transaction)

        prob = result["fraud_probability"]
        risk = result["risk_level"]

        st.divider()

        mc1, mc2 = st.columns([1, 1.2])

        with mc1:
            st.markdown("### 🔍 Risk Classification Result")
            if risk == "HIGH RISK":
                st.markdown(f"""
                <div class="risk-high" style="font-size:1.1rem; padding:12px 24px; text-align:center; margin-bottom:1.5rem;">
                    🚨 HIGH RISK DETECTED (Probability: {prob:.2%})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low" style="font-size:1.1rem; padding:12px 24px; text-align:center; margin-bottom:1.5rem;">
                    ✅ LOW RISK (Probability: {prob:.2%})
                </div>
                """, unsafe_allow_html=True)

            if result["rule_violations"]:
                st.warning("⚠️ **Hard Business Rules Flagged:**")
                for rule in result["rule_violations"]:
                    st.markdown(f"**• {rule['rule_name']}**: *{rule['description']}*")
            else:
                st.success("✅ **Passed all Hard Business Rule checks.**")

            st.markdown("### 📝 Behavioral Explanation Details")
            st.text_area("Prediction Explanation", value=result["explanation"], height=160, label_visibility="collapsed")

        with mc2:
            # Dynamic Feature Contributions Chart
            fig_xai = render_xai_chart(result["contributions"])
            st.plotly_chart(fig_xai, use_container_width=True)


# ── Tab 5: Advanced Analytics & Metrics ──────────────────────────
with tab5:
    st.markdown('<div class="section-header">📈 Live System Fraud Intelligence & Analytics</div>', unsafe_allow_html=True)

    recent_all = db.get_recent_transactions(limit=600)

    if recent_all:
        df_all = pd.DataFrame(recent_all)

        c1, c2 = st.columns(2)

        # Chart 1: Probability Distribution
        with c1:
            fig_hist = px.histogram(
                df_all, x="fraud_probability", nbins=30,
                color="risk_level",
                color_discrete_map={"HIGH RISK": "#f43f5e", "LOW RISK": "#10b981"},
                title="<b>Fraud Risk Probability Distribution</b>",
                labels={"fraud_probability": "Fraud Probability Cutoff", "count": "Count"},
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                title_font_size=15,
                bargap=0.08,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Chart 2: Hourly Fraud Distribution
        with c2:
            hourly = db.get_hourly_fraud_distribution()
            if hourly:
                df_hourly = pd.DataFrame(hourly)
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=df_hourly["hour"], y=df_hourly["total"],
                    name="Normal Volume", marker_color="rgba(99, 102, 241, 0.45)",
                ))
                fig_hourly.add_trace(go.Bar(
                    x=df_hourly["hour"], y=df_hourly["fraud_count"],
                    name="Fraud Flagged", marker_color="#f43f5e",
                ))
                fig_hourly.update_layout(
                    title="<b>Volume & Fraud Count by Hour</b>",
                    barmode="overlay",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#cbd5e1",
                    title_font_size=15,
                    xaxis_title="Hour of Day (24h)",
                    yaxis_title="Count",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

        # Row 2 Analytics
        c3, c4 = st.columns(2)

        # Chart 3: Location Fraud Breakdown
        with c3:
            location_fraud = df_all.groupby("location").agg(
                total=("transaction_id", "count"),
                high_risk=("risk_level", lambda x: (x == "HIGH RISK").sum()),
            ).reset_index()

            fig_loc = px.bar(
                location_fraud, x="location", y=["total", "high_risk"],
                barmode="group",
                color_discrete_sequence=["rgba(99, 102, 241, 0.7)", "#f43f5e"],
                title="<b>Normal vs. Fraud Volume by Location</b>",
                labels={"value": "Transaction Volume Count", "location": "Location"}
            )
            fig_loc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                title_font_size=15,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_loc, use_container_width=True)

        # Chart 4: Merchant Fraud Share
        with c4:
            merchant_fraud = df_all.groupby("merchant_id").agg(
                high_risk=("risk_level", lambda x: (x == "HIGH RISK").sum()),
            ).reset_index()

            fig_merch = px.pie(
                merchant_fraud, names="merchant_id", values="high_risk",
                title="<b>Distribution of Fraud Alerts by Merchant Gateway</b>",
                color_discrete_sequence=px.colors.sequential.RdPu_r,
                hole=0.4,
            )
            fig_merch.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                title_font_size=15,
            )
            st.plotly_chart(fig_merch, use_container_width=True)

        # Chart 5: User Risk Scores List
        st.markdown("#### 👤 Per-User Fraud Risk Aggregates (Top 20 Accounts)")
        user_risk = db.get_user_risk_summary()
        if user_risk:
            df_user = pd.DataFrame(user_risk)
            fig_user = px.bar(
                df_user, x="user_id", y="avg_risk",
                color="fraud_count",
                color_continuous_scale=["#10b981", "#f59e0b", "#f43f5e"],
                title="<b>User Profiling — Average Risk Score and Fraud Count</b>",
                labels={"avg_risk": "Avg Risk Probability", "user_id": "User Account ID",
                        "fraud_count": "Fraud Alerts Hit"},
            )
            fig_user.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                title_font_size=15,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_user, use_container_width=True)

    else:
        st.info("No analytics data compiled. Stream simulator payments to compile charts.")


# ── Auto-Refresh Execution Block ──────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(2)
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem; padding:1.2rem;'>"
    "COGNITIVE SHIELD — Real-time Financial Security Engine • Powered by Explainable ML & Advanced Behavioral Analytics"
    "</div>",
    unsafe_allow_html=True,
)
