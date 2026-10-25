"""
Visualization Utilities for RL-ACK Smart Kitchen IoT Dashboard.
Generates interactive Plotly figures for network metrics, sensors, topology, and policies.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Coordinated color palette
POLICY_COLORS = {
    "Static QoS 0": "#94a3b8",            # Slate gray
    "Static Local ACK": "#38bdf8",         # Sky blue
    "Static QoS 1": "#818cf8",             # Indigo
    "Static Local+Selective": "#f59e0b",   # Amber
    "Static QoS 2": "#f43f5e",             # Rose
    "RL-ACK DQN": "#10b981",               # Emerald green (highlight)
}

TIER_COLORS = {
    "CC": "#ef4444",   # Red (Critical Control)
    "HBS": "#3b82f6",  # Blue (Streaming)
    "BET": "#10b981",  # Green (Telemetry)
}


def create_network_topology_fig(nodes: list[dict]) -> go.Figure:
    """Create 2D star topology diagram with edge router and surrounding IoT nodes."""
    fig = go.Figure()

    if not nodes:
        fig.add_annotation(text="No topology data available", showarrow=False)
        return fig

    # Separate router and devices
    router = next((n for n in nodes if "router" in n.get("node_id", "").lower()), nodes[0])
    devices = [n for n in nodes if n != router]

    rx, ry = router.get("position", (7.5, 7.5))

    # Draw star communication links
    for dev in devices:
        dx, dy = dev.get("position", (0, 0))
        fig.add_trace(go.Scatter(
            x=[dx, rx], y=[dy, ry],
            mode="lines",
            line=dict(color="#cbd5e1", width=1.5, dash="dot"),
            hoverinfo="none",
            showlegend=False
        ))

    # Add devices
    dev_x = [d.get("position", (0, 0))[0] for d in devices]
    dev_y = [d.get("position", (0, 0))[1] for d in devices]
    dev_text = [
        f"<b>{d.get('node_id')}</b><br>Category: {d.get('category')}<br>Sensors: {', '.join(d.get('sensor_ids', []))}"
        for d in devices
    ]

    fig.add_trace(go.Scatter(
        x=dev_x, y=dev_y,
        mode="markers+text",
        marker=dict(size=18, color="#38bdf8", line=dict(color="#0284c7", width=2)),
        text=[d.get("node_id", "").replace("device_", "D") for d in devices],
        textposition="top center",
        hovertext=dev_text,
        hoverinfo="text",
        name="IoT Devices"
    ))

    # Add edge router at center
    fig.add_trace(go.Scatter(
        x=[rx], y=[ry],
        mode="markers+text",
        marker=dict(size=28, color="#10b981", symbol="diamond", line=dict(color="#047857", width=3)),
        text=["Edge Router (RL-ACK)"],
        textposition="bottom center",
        hovertext="<b>Edge Router</b><br>Hosts RL-ACK DQN Controller & Broker",
        hoverinfo="text",
        name="Edge Router"
    ))

    fig.update_layout(
        title="Smart Kitchen Star Topology (Wi-Fi 6 / 802.11ax)",
        xaxis=dict(title="X Position (m)", range=[-1, 16], zeroline=False),
        yaxis=dict(title="Y Position (m)", range=[-1, 16], zeroline=False),
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_sensor_timeseries_fig(readings: list[dict], sensor_type: str = "gas_concentration",
                                 threshold: float = 50.0, unit: str = "ppm") -> go.Figure:
    """Create interactive time series chart of sensor readings with threshold line."""
    fig = go.Figure()

    filtered = [r for r in readings if r.get('sensor_type') == sensor_type]
    if not filtered:
        fig.add_annotation(text=f"No readings recorded for {sensor_type}", showarrow=False)
        return fig

    filtered.sort(key=lambda x: x['timestamp'])
    df = pd.DataFrame(filtered)

    # Main sensor trace
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['value'],
        mode='lines+markers',
        name=f"{sensor_type.replace('_', ' ').title()}",
        line=dict(color="#ef4444" if "gas" in sensor_type or "smoke" in sensor_type else "#3b82f6", width=2.5),
        marker=dict(size=4),
        hovertemplate="Time: %{x:.1f}s<br>Reading: %{y:.2f} " + unit + "<br>Status: %{text}<extra></extra>",
        text=df['status']
    ))

    # Threshold horizontal reference line
    if threshold:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Alert Threshold ({threshold} {unit})",
            annotation_position="top right"
        )

    fig.update_layout(
        title=f"Kitchen Sensor: {sensor_type.replace('_', ' ').title()} Time Series",
        xaxis_title="Simulation Time (s)",
        yaxis_title=f"Concentration / Value ({unit})",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_latency_vs_utilization_fig(df: pd.DataFrame) -> go.Figure:
    """Plot latency versus network utilization for all policies."""
    fig = go.Figure()
    if df.empty or 'utilization_target' not in df.columns:
        fig.add_annotation(text="No experiment data available", showarrow=False)
        return fig

    for policy in df['policy'].unique():
        pdf = df[df['policy'] == policy].sort_values('utilization_target')
        color = POLICY_COLORS.get(policy, "#64748b")
        fig.add_trace(go.Scatter(
            x=pdf['utilization_target'],
            y=pdf['mean_latency_ms_mean'] if 'mean_latency_ms_mean' in pdf else pdf['mean_latency_ms'],
            mode='lines+markers',
            name=policy,
            line=dict(color=color, width=3 if policy == "RL-ACK DQN" else 1.8),
            marker=dict(size=8 if policy == "RL-ACK DQN" else 6),
        ))

    fig.update_layout(
        title="Mean End-to-End Latency vs Network Utilization",
        xaxis_title="Offered Utilization Target (ρ)",
        yaxis_title="Mean Latency (ms)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_overhead_vs_utilization_fig(df: pd.DataFrame) -> go.Figure:
    """Plot network overhead versus utilization for all policies."""
    fig = go.Figure()
    if df.empty or 'utilization_target' not in df.columns:
        fig.add_annotation(text="No experiment data available", showarrow=False)
        return fig

    for policy in df['policy'].unique():
        pdf = df[df['policy'] == policy].sort_values('utilization_target')
        color = POLICY_COLORS.get(policy, "#64748b")
        fig.add_trace(go.Scatter(
            x=pdf['utilization_target'],
            y=pdf['overhead_kb_s_mean'] if 'overhead_kb_s_mean' in pdf else pdf['overhead_kb_s'],
            mode='lines+markers',
            name=policy,
            line=dict(color=color, width=3 if policy == "RL-ACK DQN" else 1.8),
            marker=dict(size=8 if policy == "RL-ACK DQN" else 6),
        ))

    fig.update_layout(
        title="Network Overhead (KB/s) vs Network Utilization",
        xaxis_title="Offered Utilization Target (ρ)",
        yaxis_title="Overhead (KB/s)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_mdr_vs_loss_fig(df: pd.DataFrame) -> go.Figure:
    """Plot Deadline-compliant MDR versus ACK-path loss rate."""
    fig = go.Figure()
    if df.empty or 'ack_loss_rate' not in df.columns:
        fig.add_annotation(text="No experiment data available", showarrow=False)
        return fig

    for policy in df['policy'].unique():
        pdf = df[df['policy'] == policy].sort_values('ack_loss_rate')
        color = POLICY_COLORS.get(policy, "#64748b")
        mdr_col = 'mdr_overall_mean' if 'mdr_overall_mean' in pdf else 'mdr_overall'
        fig.add_trace(go.Scatter(
            x=[f"{x*100:.0f}%" for x in pdf['ack_loss_rate']],
            y=pdf[mdr_col] * 100,
            mode='lines+markers',
            name=policy,
            line=dict(color=color, width=3 if policy == "RL-ACK DQN" else 1.8),
            marker=dict(size=8 if policy == "RL-ACK DQN" else 6),
        ))

    fig.update_layout(
        title="Deadline-Compliant MDR vs ACK-Path Loss Rate",
        xaxis_title="ACK-Path Loss Rate (%)",
        yaxis_title="Deadline Compliance MDR (%)",
        template="plotly_white",
        yaxis_range=[80, 101],
        height=400,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_tier_compliance_fig(metrics: dict) -> go.Figure:
    """Bar chart comparing deadline compliance across CC, HBS, and BET."""
    mdr = metrics.get('mdr', {})
    tiers = ['Critical Control (CC)', 'High-Bandwidth (HBS)', 'Best-Effort (BET)']
    rates = [
        mdr.get('mdr_CC', 0.95) * 100,
        mdr.get('mdr_HBS', 0.90) * 100,
        mdr.get('mdr_BET', 0.99) * 100,
    ]
    colors = ['#ef4444', '#3b82f6', '#10b981']

    fig = go.Figure(go.Bar(
        x=tiers,
        y=rates,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
    ))

    fig.update_layout(
        title="Deadline-Compliant MDR by Traffic Tier",
        yaxis_title="MDR (%)",
        yaxis_range=[70, 105],
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_action_distribution_fig(action_dist: dict) -> go.Figure:
    """Grouped bar or donut chart of ACK actions applied per traffic tier."""
    if not action_dist:
        fig = go.Figure()
        fig.add_annotation(text="No action distribution data", showarrow=False)
        return fig

    action_names = ["a0 (No ACK)", "a1 (Local)", "a2 (PUBACK)", "a3 (Selective)", "a4 (QoS 2)"]
    fig = go.Figure()

    for tier, counts in action_dist.items():
        total = sum(counts.values()) or 1
        y_vals = [counts.get(f"a{i}", counts.get(str(i), 0)) / total * 100 for i in range(5)]
        fig.add_trace(go.Bar(
            name=tier,
            x=action_names,
            y=y_vals,
            marker_color=TIER_COLORS.get(tier, "#64748b"),
        ))

    fig.update_layout(
        barmode='group',
        title="ACK Action Distribution by Traffic Tier (RL-ACK)",
        xaxis_title="ACK Action",
        yaxis_title="Selection Percentage (%)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_policy_radar_fig(df_comp: pd.DataFrame) -> go.Figure:
    """Radar chart comparing multi-objective performance across all policies."""
    fig = go.Figure()
    if df_comp.empty:
        fig.add_annotation(text="No comparison data", showarrow=False)
        return fig

    categories = ['Reliability (MDR)', 'Low Latency', 'Low Overhead', 'Energy Efficiency', 'CC Protection']

    for _, row in df_comp.iterrows():
        policy = row.get('policy', 'Unknown')
        color = POLICY_COLORS.get(policy, "#64748b")

        # Normalize metrics to 0-100 scale (higher is better)
        mdr_score = float(row.get('mdr_overall', 0.9)) * 100
        lat = float(row.get('mean_latency_ms', 1.0))
        lat_score = max(0, min(100, 100 - (lat * 20)))
        ovh = float(row.get('overhead_kb_s', 0.2))
        ovh_score = max(0, min(100, 100 - (ovh * 150)))
        eng = float(row.get('energy_mj_s', 0.1))
        eng_score = max(0, min(100, 100 - (eng * 100)))
        cc_score = float(row.get('mdr_CC', 0.95)) * 100

        values = [mdr_score, lat_score, ovh_score, eng_score, cc_score]
        values.append(values[0])  # close polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself' if policy == "RL-ACK DQN" else 'none',
            name=policy,
            line=dict(color=color, width=3 if policy == "RL-ACK DQN" else 1.5),
            opacity=0.7 if policy == "RL-ACK DQN" else 0.5,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Multi-Objective Policy Tradeoff (Higher is Better)",
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig
