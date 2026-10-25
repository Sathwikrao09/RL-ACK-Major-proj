"""
Reproducible Gas Leak Demonstration Script.

Simulates a sudden kitchen gas leak event:
  1. Sensor readings remain normal (15-25 ppm).
  2. At t = 10s, a gas leak occurs, ramping concentration to ~350 ppm.
  3. Kitchen sensor detects threshold crossing (>50 ppm) and triggers simulated alert.
  4. Node switches message priority to Critical Control (CC).
  5. Compares how Static policies handle the burst vs RL-ACK adaptive ACK selection.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.network import SmartKitchenSimulation
from policies.baselines import StaticPolicy, DQNPolicy
from rl.dqn_agent import DQNAgent
from ack.actions import AckAction
from sensors.sensor_models import SensorType


def run_gas_leak_demo():
    print("=" * 75)
    print("SIMULATED SMART KITCHEN: REPRODUCIBLE GAS LEAK DEMONSTRATION")
    print("Scenario: Gas leak event triggered at t=10s | Threshold=50.0 ppm")
    print("=" * 75)

    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 30-second demonstration
    config['simulation']['duration_s'] = 30.0
    config['network']['default_device_count'] = 10
    config['network']['default_utilization'] = 0.7
    config['network']['default_ack_loss'] = 0.08  # 8% loss to demonstrate robustness

    # Load DQN agent if available
    agent = DQNAgent(config.get("dqn", {}), seed=42)
    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "dqn_checkpoint_latest.pt")
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)

    # 1. Run simulation with RL-ACK
    sim = SmartKitchenSimulation(
        config=config,
        policy=DQNPolicy(agent, deterministic=True),
        seed=42,
    )
    sim.setup()

    # Find the gas sensor and inject the leak
    gas_sensor = None
    for sensor in sim.sensors.values():
        if sensor.sensor_type == SensorType.GAS_CONCENTRATION:
            gas_sensor = sensor
            break

    if gas_sensor:
        print(f"\n[DEMO] Found gas sensor: {gas_sensor.sensor_id} on {gas_sensor.device_id}")
        gas_sensor.inject_event(
            event_type="gas_leak",
            start_time=10.0,
            duration=15.0,
            peak_value=350.0,
        )
        print("[DEMO] Injected simulated gas leak: start=10.0s, duration=15.0s, peak=350.0 ppm\n")

    # Run simulation
    results = sim.run()

    # Inspect sensor reading timeline
    print("\n" + "-" * 75)
    print("TIMELINE OF GAS SENSOR READINGS DURING SIMULATION:")
    print("-" * 75)
    print(f"{'Time (s)':<10} {'Reading (ppm)':<16} {'Status':<12} {'Alert Level':<12}")
    print("-" * 75)

    if gas_sensor:
        for r in gas_sensor.reading_history:
            if r.timestamp % 2.0 < 1.0 or r.status.name != "NORMAL":
                print(f"{r.timestamp:<10.1f} {r.value:<16.2f} {r.status.name:<12} {r.alert_level.name:<12}")

    # Inspect alerts generated
    alerts = sim.get_alert_history()
    print("\n" + "-" * 75)
    print(f"ALERTS LOGGED BY EDGE ROUTER: {len(alerts)} alert events")
    for a in alerts[:5]:
        print(f"  * [t={a['time']:.1f}s] Sensor={a['sensor_id']} | Level={a['alert_level']} | Value={a['value']:.1f} ppm | Msg: {a['message']}")
    print("-" * 75)

    print("\nNETWORK PERFORMANCE SUMMARY UNDER GAS EMERGENCY TRAFFIC:")
    print(f"  Overall MDR:             {results['mdr']['mdr_overall']*100:.2f}%")
    print(f"  Critical Control (CC) MDR: {results['mdr'].get('mdr_CC', 0)*100:.2f}%")
    print(f"  Mean CC Latency:         {results['mdr'].get('mean_latency_CC_ms', 0):.2f} ms (Deadline: 10.0 ms)")
    print(f"  Mean End-to-End Latency: {results['latency']['mean_latency_ms']:.2f} ms")
    print(f"  Network Overhead:        {results['overhead']['overhead_kb_per_s']:.2f} KB/s")
    print("=" * 75)


if __name__ == "__main__":
    run_gas_leak_demo()
