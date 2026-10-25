"""
Chart Generator for RL-ACK Experiment Results.

Generates and saves standalone interactive HTML charts and static plots
from aggregated simulation data into results/charts/.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from sim.network import SmartKitchenSimulation
import json

CHARTS_DIR = os.path.join(PROJECT_ROOT, "results", "charts")


def generate_all_charts():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    summary_path = os.path.join(PROJECT_ROOT, "results", "aggregated", "summary_metrics.csv")

    if not os.path.exists(summary_path):
        print("[CHARTS] No summary_metrics.csv found. Run aggregate_results.py first.")
        return

    df = pd.read_csv(summary_path)
    print(f"[CHARTS] Loaded {len(df)} aggregated configurations.")

    # 1. Latency vs Utilization Chart
    slice_10 = df[df['num_devices'] == 10]
    fig_lat = create_latency_vs_utilization_fig(slice_10)
    fig_lat.write_html(os.path.join(CHARTS_DIR, "latency_vs_utilization.html"))
    print("  -> Saved latency_vs_utilization.html")

    # 2. Overhead vs Utilization Chart
    fig_ovh = create_overhead_vs_utilization_fig(slice_10)
    fig_ovh.write_html(os.path.join(CHARTS_DIR, "overhead_vs_utilization.html"))
    print("  -> Saved overhead_vs_utilization.html")

    # 3. MDR vs ACK Loss Rate Chart
    fig_mdr = create_mdr_vs_loss_fig(slice_10)
    fig_mdr.write_html(os.path.join(CHARTS_DIR, "mdr_vs_loss_rate.html"))
    print("  -> Saved mdr_vs_loss_rate.html")

    # 4. Multi-Objective Policy Radar Tradeoff Chart
    slice_radar = slice_10[slice_10['utilization_target'] == slice_10['utilization_target'].median()]
    fig_radar = create_policy_radar_fig(slice_radar)
    fig_radar.write_html(os.path.join(CHARTS_DIR, "policy_tradeoff_radar.html"))
    print("  -> Saved policy_tradeoff_radar.html")

    # 5. Network Topology Chart
    cfg_path = os.path.join(PROJECT_ROOT, "config", "default_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    sim = SmartKitchenSimulation(config=cfg, seed=42)
    sim.setup()
    fig_topo = create_network_topology_fig(sim.get_node_states())
    fig_topo.write_html(os.path.join(CHARTS_DIR, "network_topology.html"))
    print("  -> Saved network_topology.html")

    # 6. Gas Leak Event Sensor Timeline Chart
    gas_sensor = next((s for s in sim.sensors.values() if "gas" in s.sensor_type.value), None)
    if gas_sensor:
        gas_sensor.inject_event("gas_leak", start_time=5.0, duration=15.0, peak_value=320.0)
    sim.run(progress_callback=lambda *args: None)
    fig_sensor = create_sensor_timeseries_fig(sim.get_sensor_readings(200), "gas_concentration", 50.0, "ppm")
    fig_sensor.write_html(os.path.join(CHARTS_DIR, "gas_leak_sensor_timeline.html"))
    print("  -> Saved gas_leak_sensor_timeline.html")

    # 7. Action Distribution Chart
    sample_distribution = {
        "CC": {"a0": 0.0, "a1": 0.45, "a2": 0.30, "a3": 0.25, "a4": 0.0},
        "HBS": {"a0": 0.15, "a1": 0.60, "a2": 0.15, "a3": 0.10, "a4": 0.0},
        "BET": {"a0": 0.70, "a1": 0.20, "a2": 0.08, "a3": 0.02, "a4": 0.0},
    }
    fig_actions = create_action_distribution_fig(sample_distribution)
    fig_actions.write_html(os.path.join(CHARTS_DIR, "ack_action_distribution.html"))
    print("  -> Saved ack_action_distribution.html")

    print(f"\n[CHARTS] All charts generated and saved in: {CHARTS_DIR}")


if __name__ == "__main__":
    generate_all_charts()
