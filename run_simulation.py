#!/usr/bin/env python3
"""
RL-ACK Smart Kitchen IoT Simulator — Manual CLI Simulation Runner.

Allows running complete, standalone packet-level discrete-event simulations
directly from the terminal with full parameter control, live logging,
per-tier delivery analysis, and export to CSV/JSON.

Usage Examples:
  # Quick 30-second run with default RL-ACK policy:
  python run_simulation.py --duration 30 --devices 10 --policy rl-ack

  # Stress test under 80% utilization and 10% ACK-path packet loss:
  python run_simulation.py --duration 60 --devices 20 --utilization 0.8 --ack-loss 0.10 --policy rl-ack

  # Compare against Static QoS 1:
  python run_simulation.py --duration 30 --devices 10 --policy static_a2

  # Emergency scenario with simulated gas leak at t=10s:
  python run_simulation.py --duration 30 --devices 10 --inject-gas-leak --policy rl-ack

  # Interactive prompt mode:
  python run_simulation.py --interactive
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.network import SmartKitchenSimulation
from policies.baselines import get_all_policies, StaticPolicy, DQNPolicy, get_policy_by_name
from rl.dqn_agent import DQNAgent
from ack.actions import AckAction
from sensors.sensor_models import SensorType


def load_config() -> dict:
    cfg_path = os.path.join(PROJECT_ROOT, "config", "default_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_policy(policy_name: str, agent: DQNAgent):
    """Resolve CLI policy string to Policy instance."""
    name_clean = policy_name.lower().replace("-", "_").replace(" ", "_")

    mapping = {
        "rl_ack": lambda: DQNPolicy(agent, deterministic=True),
        "dqn": lambda: DQNPolicy(agent, deterministic=True),
        "static_a0": lambda: StaticPolicy(AckAction.A0_NO_ACK),
        "static_qos_0": lambda: StaticPolicy(AckAction.A0_NO_ACK),
        "static_a1": lambda: StaticPolicy(AckAction.A1_LOCAL_ACK),
        "static_local_ack": lambda: StaticPolicy(AckAction.A1_LOCAL_ACK),
        "static_a2": lambda: StaticPolicy(AckAction.A2_PUBACK),
        "static_qos_1": lambda: StaticPolicy(AckAction.A2_PUBACK),
        "static_a3": lambda: StaticPolicy(AckAction.A3_LOCAL_SELECTIVE),
        "static_local_selective": lambda: StaticPolicy(AckAction.A3_LOCAL_SELECTIVE),
        "static_a4": lambda: StaticPolicy(AckAction.A4_QOS2_HANDSHAKE),
        "static_qos_2": lambda: StaticPolicy(AckAction.A4_QOS2_HANDSHAKE),
    }

    if name_clean in mapping:
        return mapping[name_clean]()

    found = get_policy_by_name(policy_name, agent)
    if found:
        return found

    print(f"[WARN] Unknown policy '{policy_name}', defaulting to RL-ACK DQN.")
    return DQNPolicy(agent, deterministic=True)


def run_interactive_mode():
    """Prompt the user step-by-step for simulation parameters."""
    print("\n" + "=" * 65)
    print("   RL-ACK SMART KITCHEN IoT SIMULATOR — MANUAL CONFIGURATION")
    print("=" * 65)

    duration_str = input("Simulation Duration in seconds [default: 30.0]: ").strip()
    duration = float(duration_str) if duration_str else 30.0

    devices_str = input("Number of IoT Devices (10, 20, 30) [default: 10]: ").strip()
    devices = int(devices_str) if devices_str else 10

    util_str = input("Network Offered Utilization (0.5 to 0.9) [default: 0.70]: ").strip()
    utilization = float(util_str) if util_str else 0.70

    loss_str = input("ACK-Path Packet Loss Rate (e.g. 0.05 for 5%) [default: 0.05]: ").strip()
    ack_loss = float(loss_str) if loss_str else 0.05

    print("\nSelect Acknowledgment Policy:")
    print("  1. RL-ACK DQN (Adaptive Reinforcement Learning)")
    print("  2. Static QoS 0 (Fire-and-forget, a0)")
    print("  3. Static Local ACK (Edge proxy ACK, a1)")
    print("  4. Static QoS 1 (PUBACK with retransmission, a2)")
    print("  5. Static Local ACK + Selective Retransmit (a3)")
    print("  6. Static QoS 2 (4-step Handshake, a4)")
    pol_choice = input("Choice [1-6, default: 1]: ").strip()

    pol_map = {
        "1": "rl-ack",
        "2": "static_a0",
        "3": "static_a1",
        "4": "static_a2",
        "5": "static_a3",
        "6": "static_a4",
    }
    policy_name = pol_map.get(pol_choice, "rl-ack")

    gas_leak_prompt = input("Inject simulated gas leak event? (y/N) [default: n]: ").strip().lower()
    inject_gas_leak = gas_leak_prompt in ("y", "yes")

    return {
        "duration": duration,
        "devices": devices,
        "utilization": utilization,
        "ack_loss": ack_loss,
        "fwd_loss": 0.01,
        "policy": policy_name,
        "seed": 42,
        "inject_gas_leak": inject_gas_leak,
        "leak_start": 10.0,
        "leak_duration": 15.0,
        "leak_ppm": 350.0,
        "verbose": False,
        "output_dir": os.path.join(PROJECT_ROOT, "results", "manual_runs"),
    }


def execute_simulation(args_dict: dict):
    duration = args_dict["duration"]
    devices = args_dict["devices"]
    utilization = args_dict["utilization"]
    ack_loss = args_dict["ack_loss"]
    fwd_loss = args_dict["fwd_loss"]
    policy_str = args_dict["policy"]
    seed = args_dict["seed"]
    inject_leak = args_dict["inject_gas_leak"]
    leak_start = args_dict.get("leak_start", 10.0)
    leak_dur = args_dict.get("leak_duration", 15.0)
    leak_ppm = args_dict.get("leak_ppm", 350.0)
    output_dir = args_dict.get("output_dir", os.path.join(PROJECT_ROOT, "results", "manual_runs"))

    config = load_config()
    config['simulation']['duration_s'] = duration
    config['network']['default_device_count'] = devices
    config['network']['default_utilization'] = utilization
    config['network']['default_ack_loss'] = ack_loss
    config['network']['default_fwd_loss'] = fwd_loss

    # Load or initialize DQN agent
    agent = DQNAgent(config.get("dqn", {}), seed=seed)
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "dqn_checkpoint_latest.pt")
    if os.path.exists(ckpt_path):
        agent.load_checkpoint(ckpt_path)

    policy = resolve_policy(policy_str, agent)

    print("\n" + "=" * 75)
    print("RL-ACK SMART KITCHEN SIMULATION EXECUTION")
    print("=" * 75)
    print(f"  Duration:            {duration:.1f} s")
    print(f"  IoT Device Count:    {devices} nodes (Star Topology, Wi-Fi 6)")
    print(f"  Offered Utilization: {utilization*100:.1f} %")
    print(f"  ACK-Path Loss Rate:  {ack_loss*100:.1f} %")
    print(f"  Forward Loss Rate:   {fwd_loss*100:.1f} %")
    print(f"  Policy:              {policy.name}")
    print(f"  Random Seed:         {seed}")
    print(f"  Gas Leak Simulated:  {'YES (at t=' + str(leak_start) + 's, ' + str(leak_ppm) + ' ppm)' if inject_leak else 'NO'}")
    print("=" * 75)

    sim = SmartKitchenSimulation(
        config=config,
        policy=policy,
        dqn_agent=agent if isinstance(policy, DQNPolicy) else None,
        seed=seed,
    )
    sim.setup()

    # Inject gas leak if requested
    if inject_leak:
        for s in sim.sensors.values():
            if s.sensor_type == SensorType.GAS_CONCENTRATION:
                s.inject_event("gas_leak", start_time=leak_start, duration=leak_dur, peak_value=leak_ppm)
                print(f"[EVENT] Injected gas leak on {s.sensor_id} ({s.device_id}) at t={leak_start:.1f}s")
                break

    # Run simulation with progress output
    print(f"\n[RUN] Simulating packet traffic for {duration:.1f}s...\n")
    start_time = time.time()
    results = sim.run()
    wall_elapsed = time.time() - start_time

    # Output Detailed Console Report
    mdr = results['mdr']
    lat = results['latency']
    ovh = results['overhead']
    eng = results['energy']
    engine = results['engine_stats']

    print("\n" + "=" * 75)
    print("                   SIMULATION RESULTS REPORT")
    print("=" * 75)

    print("ENGINE EXECUTION:")
    print(f"  Discrete Events Processed: {engine['total_events_processed']:,}")
    print(f"  Simulation Virtual Clock:  {engine['sim_time_reached']:.2f} s")
    print(f"  Execution Wall Time:       {wall_elapsed:.2f} s ({engine['total_events_processed']/max(wall_elapsed, 0.001):,.0f} events/s)")

    print("\nMESSAGE DELIVERY RATIO (MDR):")
    print(f"  Total Messages Generated:  {mdr['total_messages']:,}")
    print(f"  Total Delivered On-Time:   {mdr['delivered_count']:,} ({mdr['delivery_rate']*100:.2f}% delivery rate)")
    print(f"  Overall Deadline-MDR:      {mdr['mdr_overall']*100:.2f} %")
    print(f"    - Critical Control (CC): {mdr.get('mdr_CC', 0)*100:.2f} % (Deadline: 10 ms)")
    print(f"    - Streaming (HBS):       {mdr.get('mdr_HBS', 0)*100:.2f} % (Deadline: 50 ms)")
    print(f"    - Telemetry (BET):       {mdr.get('mdr_BET', 0)*100:.2f} % (Deadline: 500 ms)")

    print("\nLATENCY PERFORMANCE:")
    print(f"  Mean End-to-End Latency:   {lat['mean_latency_ms']:.2f} ms")
    print(f"  Median Latency:            {lat['median_latency_ms']:.2f} ms")
    print(f"  95th Percentile Latency:   {lat['p95_latency_ms']:.2f} ms")
    print(f"  Latency Jitter:            {lat['jitter_ms']:.2f} ms")
    print(f"  Mean CC Latency:           {mdr.get('mean_latency_CC_ms', 0):.2f} ms")
    print(f"  Mean HBS Latency:          {mdr.get('mean_latency_HBS_ms', 0):.2f} ms")
    print(f"  Mean BET Latency:          {mdr.get('mean_latency_BET_ms', 0):.2f} ms")

    print("\nNETWORK OVERHEAD & ENERGY:")
    print(f"  Network Overhead Rate:     {ovh['overhead_kb_per_s']:.2f} KB/s")
    print(f"  Total Overhead Transferred:{ovh['total_overhead_bytes']:,} Bytes")
    print(f"  Control-Plane Overhead:    {ovh['control_plane_overhead_pct']:.2f} %")
    print(f"  Modeled Radio Energy:      {eng['energy_mj_per_s']:.2f} mJ/s ({eng['total_energy_mj']:.1f} mJ total)")

    if results.get('action_distribution'):
        print("\nACK ACTION DISTRIBUTION (RL-ACK DQN):")
        dist = results['action_distribution']
        action_names = ["a0(No ACK)", "a1(Local)", "a2(PUBACK)", "a3(Selective)", "a4(QoS 2)"]
        for tier in ['CC', 'HBS', 'BET']:
            if tier in dist:
                tier_counts = dist[tier]
                total = sum(tier_counts.values()) or 1
                row_str = " | ".join(f"{name}: {tier_counts.get(f'a{i}', 0)/total*100:4.1f}%" for i, name in enumerate(action_names))
                print(f"  Tier {tier:<3} -> {row_str}")

    # Alerts logged
    alerts = sim.get_alert_history()
    if alerts:
        print(f"\nSAFETY ALERTS DETECTED DURING RUN ({len(alerts)} events):")
        for a in alerts[:5]:
            print(f"  * [t={a['time']:5.1f}s] Sensor: {a['sensor_id']} | Value: {a['value']:5.1f} | Level: {a['alert_level']} | {a['message']}")
        if len(alerts) > 5:
            print(f"  ... and {len(alerts) - 5} more alert events logged.")
    else:
        print("\nSAFETY STATUS: All sensor readings nominal. Zero alert threshold crossings.")

    print("=" * 75)

    # Save Run Outputs
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp = int(time.time())
    policy_tag = policy.name.lower().replace(" ", "_")
    csv_file = os.path.join(output_dir, f"run_{run_timestamp}_{policy_tag}_packets.csv")
    json_file = os.path.join(output_dir, f"run_{run_timestamp}_{policy_tag}_summary.json")

    sim.metrics.export_csv(csv_file)
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    print(f"\n[OUTPUT] Saved per-message packet log: {csv_file}")
    print(f"[OUTPUT] Saved summary metrics JSON:  {json_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Manual Simulation Runner for RL-ACK Smart Kitchen")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive parameter questionnaire")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds (default: 30.0)")
    parser.add_argument("--devices", type=int, default=10, help="Number of IoT devices: 10, 20, 30 (default: 10)")
    parser.add_argument("--utilization", type=float, default=0.70, help="Offered load utilization 0.5-0.9 (default: 0.70)")
    parser.add_argument("--ack-loss", type=float, default=0.05, help="ACK-path loss rate 0.01-0.15 (default: 0.05)")
    parser.add_argument("--fwd-loss", type=float, default=0.01, help="Forward loss rate (default: 0.01)")
    parser.add_argument("--policy", type=str, default="rl-ack",
                        help="Policy: rl-ack, static_a0, static_a1, static_a2, static_a3, static_a4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--inject-gas-leak", action="store_true", help="Simulate a gas leak emergency event")
    parser.add_argument("--leak-start", type=float, default=10.0, help="Gas leak start time in seconds (default: 10.0)")
    parser.add_argument("--leak-duration", type=float, default=15.0, help="Gas leak duration in seconds (default: 15.0)")
    parser.add_argument("--leak-ppm", type=float, default=350.0, help="Peak gas concentration in ppm (default: 350.0)")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory for logs")

    args = parser.parse_args()

    if args.interactive:
        run_args = run_interactive_mode()
    else:
        run_args = {
            "duration": args.duration,
            "devices": args.devices,
            "utilization": args.utilization,
            "ack_loss": args.ack_loss,
            "fwd_loss": args.fwd_loss,
            "policy": args.policy,
            "seed": args.seed,
            "inject_gas_leak": args.inject_gas_leak,
            "leak_start": args.leak_start,
            "leak_duration": args.leak_duration,
            "leak_ppm": args.leak_ppm,
            "output_dir": args.output_dir if args.output_dir else os.path.join(PROJECT_ROOT, "results", "manual_runs"),
        }

    execute_simulation(run_args)


if __name__ == "__main__":
    main()
