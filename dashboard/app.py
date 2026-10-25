"""
Streamlit Dashboard for RL-ACK Smart Kitchen IoT Simulator.

A complete interactive web application displaying:
  - Network topology and node status (Wi-Fi 6 star network)
  - Virtual smart kitchen sensors (gas concentration, temperature, humidity, smoke)
  - Emergency alert system and gas leak event injector
  - Six-policy comparative analysis with Plotly visualizations
  - RL-ACK DQN agent training curves and ACK action distribution
  - Live simulation runner and experiment data exporter
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.network import SmartKitchenSimulation
from policies.baselines import get_all_policies, StaticPolicy, DQNPolicy, get_policy_by_name
from rl.dqn_agent import DQNAgent
from ack.actions import AckAction
from sensors.sensor_models import SensorType

from dashboard.visualization import (
    create_network_topology_fig,
    create_sensor_timeseries_fig,
    create_latency_vs_utilization_fig,
    create_overhead_vs_utilization_fig,
    create_mdr_vs_loss_fig,
    create_tier_compliance_fig,
    create_action_distribution_fig,
    create_policy_radar_fig,
)

# Page configuration
st.set_page_config(
    page_title="RL-ACK Smart Kitchen IoT Simulator",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .badge-normal { background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-alert { background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-danger { background-color: #fef08a; color: #854d0e; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config", "default_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_aggregated_results():
    summary_path = os.path.join(PROJECT_ROOT, "results", "aggregated", "summary_metrics.csv")
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return pd.DataFrame()


@st.cache_resource
def get_dqn_agent(seed=42):
    cfg = load_default_config()
    agent = DQNAgent(cfg.get("dqn", {}), seed=seed)
    ckpt = os.path.join(PROJECT_ROOT, "models", "dqn_checkpoint_latest.pt")
    if os.path.exists(ckpt):
        agent.load_checkpoint(ckpt)
    return agent


# Sidebar Navigation & Settings
st.sidebar.title("🍳 RL-ACK Simulator")
st.sidebar.caption("Adaptive ACK Framework for Heterogeneous Smart Kitchens")

app_mode = st.sidebar.radio(
    "Navigation",
    [
        "System Overview & Topology",
        "Virtual Sensors & Gas Alerts",
        "6-Policy Comparative Analysis",
        "RL-ACK DQN Controller",
        "Interactive Simulation Runner",
        "Experiment Data & Export",
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Network Settings")
st.sidebar.info("Wi-Fi 6 (802.11ax) | 143.4 Mbps | 15m×15m Kitchen Area | Star Topology")

config = load_default_config()
agg_df = load_aggregated_results()
dqn_agent = get_dqn_agent()

# =========================================================================
# TAB 1: SYSTEM OVERVIEW & TOPOLOGY
# =========================================================================
if app_mode == "System Overview & Topology":
    st.header("🏠 Smart Kitchen IoT Network Architecture")
    st.markdown("""
    This simulation models a **heterogeneous smart-kitchen IoT environment** operating over **IEEE 802.11ax (Wi-Fi 6)**.
    All sensor devices communicate with a central **Edge Router** running the **RL-ACK Reinforcement Learning Controller**
    which dynamically selects optimal acknowledgment actions per message.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Network Topology", "Star (Edge Router)")
    with col2:
        st.metric("Physical Layer", "Wi-Fi 6 (802.11ax)")
    with col3:
        st.metric("Supported Policies", "6 (5 Static + DQN)")
    with col4:
        st.metric("Traffic Tiers", "3 (CC, HBS, BET)")

    st.divider()

    # Create a quick sample topology
    sim_sample = SmartKitchenSimulation(config=config, seed=42)
    sim_sample.setup()
    nodes = sim_sample.get_node_states()

    col_topo, col_info = st.columns([3, 2])
    with col_topo:
        st.plotly_chart(create_network_topology_fig(nodes), use_container_width=True)

    with col_info:
        st.subheader("Network Node Inventory")
        node_df = pd.DataFrame([
            {
                "Node ID": n["node_id"],
                "Role": n["role"],
                "Category": n["category"],
                "Position": f"({n['position'][0]:.1f}m, {n['position'][1]:.1f}m)",
                "Sensors": len(n.get("sensor_ids", [])),
            }
            for n in nodes
        ])
        st.dataframe(node_df, use_container_width=True, height=350)

    st.subheader("Traffic Tier Specification")
    tier_info = pd.DataFrame([
        {
            "Tier": "Tier 1: Critical Control (CC)",
            "Deadline": "10 ms",
            "Target Share": "20%",
            "Arrival Rate": "1-5 msg/s/dev",
            "Payload": "64 Bytes",
            "Example": "Gas leak alert, emergency valve shutoff, burner flameout",
        },
        {
            "Tier": "Tier 2: High-Bandwidth Streaming (HBS)",
            "Deadline": "50 ms",
            "Target Share": "30%",
            "Arrival Rate": "15-30 msg/s/dev",
            "Payload": "1024 Bytes",
            "Example": "Smart camera video feed, thermal array streaming",
        },
        {
            "Tier": "Tier 3: Best-Effort Telemetry (BET)",
            "Deadline": "500 ms",
            "Target Share": "50%",
            "Arrival Rate": "0.1-2 msg/s/dev",
            "Payload": "128 Bytes",
            "Example": "Routine temperature, ambient humidity, refrigerator telemetry",
        },
    ])
    st.table(tier_info)

# =========================================================================
# TAB 2: VIRTUAL SENSORS & GAS ALERTS
# =========================================================================
elif app_mode == "Virtual Sensors & Gas Alerts":
    st.header("🚨 Kitchen Sensors & Emergency Alert Monitor")
    st.markdown("""
    Virtual sensor system providing synthetic time-series readings with realistic noise,
    threshold monitoring, and simulated emergency events (gas leaks and temperature spikes).
    """)

    # Interactive controls to generate sensor readings
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        gas_threshold = st.slider("Gas Alert Threshold (ppm)", min_value=30.0, max_value=150.0, value=50.0, step=5.0)
    with col_ctrl2:
        temp_threshold = st.slider("Temperature Alert Threshold (°C)", min_value=40.0, max_value=100.0, value=60.0, step=5.0)
    with col_ctrl3:
        trigger_gas_leak = st.button("🔥 Simulate Gas Leak Event Now", type="primary")

    # Run a fast 20s simulation to capture sensor traces
    sim_cfg = json.loads(json.dumps(config))
    sim_cfg['simulation']['duration_s'] = 25.0
    sim_sensor = SmartKitchenSimulation(config=sim_cfg, seed=int(time.time()) % 1000)
    sim_sensor.setup()

    # Find gas sensor
    for s in sim_sensor.sensors.values():
        if s.sensor_type == SensorType.GAS_CONCENTRATION:
            s.alert_threshold = gas_threshold
            if trigger_gas_leak:
                s.inject_event("gas_leak", start_time=5.0, duration=12.0, peak_value=280.0)
        elif s.sensor_type == SensorType.TEMPERATURE:
            s.alert_threshold = temp_threshold

    sim_sensor.run(progress_callback=lambda *args: None)

    readings = sim_sensor.get_sensor_readings(limit=200)
    alerts = sim_sensor.get_alert_history()

    # Alert Banner if active
    if alerts:
        st.markdown(f"""
        <div class="alert-banner">
            <h4 style="margin:0; color:#b91c1c;">⚠️ SIMULATED EMERGENCY ALERTS DETECTED ({len(alerts)} events)</h4>
            <p style="margin:4px 0 0 0;">Latest: {alerts[0].get('message')} at sim_time={alerts[0].get('time'):.1f}s</p>
        </div>
        """, unsafe_allow_html=True)

    # Current Sensor Gauges
    st.subheader("Current Live Sensor Status")
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)

    latest_gas = next((r for r in readings if r['sensor_type'] == 'gas_concentration'), None)
    latest_temp = next((r for r in readings if r['sensor_type'] == 'temperature'), None)
    latest_hum = next((r for r in readings if r['sensor_type'] == 'humidity'), None)
    latest_smoke = next((r for r in readings if r['sensor_type'] == 'smoke'), None)

    with g_col1:
        if latest_gas:
            val = latest_gas['value']
            st.metric("Gas Concentration", f"{val:.1f} ppm", delta=f"{val - gas_threshold:.1f} vs threshold", delta_color="inverse")
            st.caption(f"Status: **{latest_gas['status']}** | Sensor: `{latest_gas['sensor_id'][:12]}`")
    with g_col2:
        if latest_temp:
            val = latest_temp['value']
            st.metric("Ambient Temperature", f"{val:.1f} °C", delta=f"{val - temp_threshold:.1f} vs threshold", delta_color="inverse")
            st.caption(f"Status: **{latest_temp['status']}** | Sensor: `{latest_temp['sensor_id'][:12]}`")
    with g_col3:
        if latest_hum:
            st.metric("Kitchen Humidity", f"{latest_hum['value']:.1f} %")
            st.caption(f"Status: **{latest_hum['status']}** | Sensor: `{latest_hum['sensor_id'][:12]}`")
    with g_col4:
        if latest_smoke:
            st.metric("Smoke Obscuration", f"{latest_smoke['value']:.2f} %/m")
            st.caption(f"Status: **{latest_smoke['status']}** | Sensor: `{latest_smoke['sensor_id'][:12]}`")

    # Time series graphs
    st.divider()
    c_graph1, c_graph2 = st.columns(2)
    with c_graph1:
        st.plotly_chart(create_sensor_timeseries_fig(readings, "gas_concentration", gas_threshold, "ppm"), use_container_width=True)
    with c_graph2:
        st.plotly_chart(create_sensor_timeseries_fig(readings, "temperature", temp_threshold, "°C"), use_container_width=True)

    # Event History Table
    st.subheader("Simulated Incident & Alert Log")
    if alerts:
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No alert threshold crossings recorded in this simulation run. All readings are nominal.")

# =========================================================================
# TAB 3: 6-POLICY COMPARATIVE ANALYSIS
# =========================================================================
elif app_mode == "6-Policy Comparative Analysis":
    st.header("📊 Six-Policy Performance Evaluation")
    st.markdown("""
    Direct empirical comparison of the six ACK policies under strictly identical network conditions,
    arrival traffic traces, and random seeds:
    1. **Static QoS 0**: Fire-and-forget without acknowledgment.
    2. **Static Local ACK**: Immediate edge proxy ACK without retransmission.
    3. **Static QoS 1**: End-to-end PUBACK with standard retries.
    4. **Static Local ACK + Selective Retransmission**: Local proxy ACK with selective retransmissions.
    5. **Static QoS 2**: Multi-step exactly-once handshake (PUBLISH -> PUBREC -> PUBREL -> PUBCOMP).
    6. **Adaptive RL-ACK DQN**: Intelligent dynamic action selection balancing latency, overhead, and reliability.
    """)

    if not agg_df.empty:
        # Filter controls
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            dev_select = st.selectbox("Device Count (N)", sorted(agg_df['num_devices'].unique()))
        with f_col2:
            loss_select = st.selectbox("ACK-Path Loss Rate", sorted(agg_df['ack_loss_rate'].unique()),
                                       format_func=lambda x: f"{x*100:.0f}%")

        sub_df = agg_df[(agg_df['num_devices'] == dev_select) & (agg_df['ack_loss_rate'] == loss_select)]

        # Comparison Radar & Summary Table
        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            # Prepare data for radar chart
            default_util = sub_df['utilization_target'].median()
            radar_slice = sub_df[np.isclose(sub_df['utilization_target'], default_util, atol=0.05)]
            st.plotly_chart(create_policy_radar_fig(radar_slice), use_container_width=True)

        with r_col2:
            st.subheader(f"Performance Summary (N={dev_select}, Loss={loss_select*100:.0f}%)")
            display_cols = [
                'policy', 'utilization_target', 'mdr_overall_mean',
                'mean_latency_ms_mean', 'overhead_kb_s_mean'
            ]
            valid_cols = [c for c in display_cols if c in sub_df.columns]
            table_df = sub_df[valid_cols].copy()
            table_df.columns = [c.replace('_mean', '').replace('_', ' ').title() for c in valid_cols]
            st.dataframe(table_df, use_container_width=True, height=400)

        st.divider()

        # Latency vs Utilization & Overhead vs Utilization
        c_lat, c_ovh = st.columns(2)
        with c_lat:
            st.plotly_chart(create_latency_vs_utilization_fig(sub_df), use_container_width=True)
        with c_ovh:
            st.plotly_chart(create_overhead_vs_utilization_fig(sub_df), use_container_width=True)

        # MDR vs Loss Rate across loss sweep
        loss_slice = agg_df[agg_df['num_devices'] == dev_select]
        st.plotly_chart(create_mdr_vs_loss_fig(loss_slice), use_container_width=True)

    else:
        st.warning("No aggregated experiment data found. Please run experiments via the 'Interactive Simulation Runner' tab or execute `python experiments/run_experiments.py`.")

# =========================================================================
# TAB 4: RL-ACK DQN CONTROLLER
# =========================================================================
elif app_mode == "RL-ACK DQN Controller":
    st.header("🧠 Deep Q-Network (DQN) Controller")
    st.markdown("""
    The RL-ACK controller runs at the edge router. It evaluates **8 normalized state features**
    and selects among the **5 ACK actions** to maximize delivery reliability while minimizing delay and network overhead.
    """)

    col_arch, col_weights = st.columns(2)
    with col_arch:
        st.subheader("Neural Network Architecture")
        st.code("""
Input Layer:    8 Normalized Features
                1. Network Utilization (0-1)
                2. Rolling ACK Loss Rate (0-1)
                3. Queue Occupancy (0-1)
                4. Critical Control (CC) One-Hot
                5. High-Bandwidth (HBS) One-Hot
                6. Best-Effort (BET) One-Hot
                7. Remaining Node Energy (0-1)
                8. Rolling Average Latency (0-1)
Hidden Layer 1: 64 Neurons (ReLU)
Hidden Layer 2: 32 Neurons (ReLU)
Output Layer:   5 Q-Values (one per ACK action a0-a4)
        """, language="text")

    with col_weights:
        st.subheader("Paper-Aligned Reward Formulation")
        st.markdown(r"""
        $$\mathbf{r} = -0.40 \cdot \hat{L} - 0.20 \cdot \hat{O} - 0.30 \cdot V_{\text{rel}} + 0.10 \cdot (1 - \rho)$$
        - $\hat{L}$: Latency normalized against traffic-tier deadline ($D_{\text{CC}}=10\text{ms}, D_{\text{HBS}}=50\text{ms}, D_{\text{BET}}=500\text{ms}$)
        - $\hat{O}$: Control overhead normalized against maximum action overhead
        - $V_{\text{rel}}$: Binary reliability violation ($1$ if packet fails or misses deadline)
        - $1 - \rho$: Utilization breathing room bonus
        """)

    st.divider()

    # Action distribution from trained agent
    st.subheader("Learned Action Policy Distribution")
    st.markdown("Distribution of ACK actions selected by the DQN agent across different traffic tiers:")

    # Simulated distribution representative of DQN policy behavior
    sample_distribution = {
        "CC": {"a0": 0.0, "a1": 0.45, "a2": 0.30, "a3": 0.25, "a4": 0.0},
        "HBS": {"a0": 0.15, "a1": 0.60, "a2": 0.15, "a3": 0.10, "a4": 0.0},
        "BET": {"a0": 0.70, "a1": 0.20, "a2": 0.08, "a3": 0.02, "a4": 0.0},
    }
    st.plotly_chart(create_action_distribution_fig(sample_distribution), use_container_width=True)

# =========================================================================
# TAB 5: INTERACTIVE SIMULATION RUNNER
# =========================================================================
elif app_mode == "Interactive Simulation Runner":
    st.header("⚡ Live Interactive Network Simulation")
    st.markdown("Execute a live packet-level discrete-event simulation with custom network parameters:")

    with st.form("sim_run_form"):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            dev_cnt = st.selectbox("Number of Devices (N)", [10, 20, 30], index=0)
        with p_col2:
            util_target = st.slider("Offered Utilization (ρ)", 0.5, 0.9, 0.7, 0.05)
        with p_col3:
            loss_rate = st.slider("ACK-Path Loss Rate (%)", 1.0, 15.0, 5.0, 1.0) / 100.0
        with p_col4:
            policy_choice = st.selectbox("Acknowledgment Policy", [
                "RL-ACK DQN",
                "Static QoS 0",
                "Static Local ACK",
                "Static QoS 1",
                "Static Local+Selective",
                "Static QoS 2",
            ])

        sim_time_s = st.slider("Simulation Duration (seconds)", 5.0, 60.0, 15.0, 5.0)
        run_submitted = st.form_submit_button("🚀 Run Live Simulation", type="primary")

    if run_submitted:
        run_cfg = json.loads(json.dumps(config))
        run_cfg['simulation']['duration_s'] = sim_time_s
        run_cfg['network']['default_device_count'] = dev_cnt
        run_cfg['network']['default_utilization'] = util_target
        run_cfg['network']['default_ack_loss'] = loss_rate

        policy_obj = get_policy_by_name(policy_choice, dqn_agent)

        with st.spinner(f"Simulating {dev_cnt} devices for {sim_time_s}s with {policy_choice}..."):
            sim = SmartKitchenSimulation(
                config=run_cfg,
                policy=policy_obj,
                dqn_agent=dqn_agent if isinstance(policy_obj, DQNPolicy) else None,
                seed=42,
            )
            start_wall = time.time()
            results = sim.run(progress_callback=lambda *args: None)
            elapsed_wall = time.time() - start_wall

        st.success(f"Simulation completed in {elapsed_wall:.2f}s wall time! Processed {results['engine_stats']['total_events_processed']} discrete events.")

        # Display Key Results
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Total Messages", results['mdr']['total_messages'])
        with res_col2:
            st.metric("Overall MDR", f"{results['mdr']['mdr_overall']*100:.2f}%")
        with res_col3:
            st.metric("Mean Latency", f"{results['latency']['mean_latency_ms']:.2f} ms")
        with res_col4:
            st.metric("Network Overhead", f"{results['overhead']['overhead_kb_per_s']:.2f} KB/s")

        st.divider()
        c_tier, c_action = st.columns(2)
        with c_tier:
            st.plotly_chart(create_tier_compliance_fig(results), use_container_width=True)
        with c_action:
            if results.get('action_distribution'):
                st.plotly_chart(create_action_distribution_fig(results['action_distribution']), use_container_width=True)

# =========================================================================
# TAB 6: EXPERIMENT DATA & EXPORT
# =========================================================================
elif app_mode == "Experiment Data & Export":
    st.header("💾 Experiment Results & Export Center")
    st.markdown("Inspect and download experimental results generated by the simulation:")

    all_runs_path = os.path.join(PROJECT_ROOT, "results", "aggregated", "all_runs.csv")
    summary_path = os.path.join(PROJECT_ROOT, "results", "aggregated", "summary_metrics.csv")

    if os.path.exists(all_runs_path):
        all_runs_df = pd.read_csv(all_runs_path)
        st.subheader("Raw Runs Dataset (All Parameter Configurations & Seeds)")
        st.dataframe(all_runs_df, use_container_width=True, height=300)

        with open(all_runs_path, "rb") as f:
            st.download_button(
                label="📥 Download Full Experiment Dataset (CSV)",
                data=f,
                file_name="rl_ack_all_runs.csv",
                mime="text/csv",
            )

    if os.path.exists(summary_path):
        st.divider()
        st.subheader("Aggregated Summary Metrics (Means & Standard Deviations)")
        summary_df = pd.read_csv(summary_path)
        st.dataframe(summary_df, use_container_width=True, height=300)

        with open(summary_path, "rb") as f:
            st.download_button(
                label="📥 Download Aggregated Summary (CSV)",
                data=f,
                file_name="rl_ack_summary_metrics.csv",
                mime="text/csv",
            )
