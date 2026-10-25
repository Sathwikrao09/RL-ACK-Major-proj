"""
Model Evaluation and Policy Comparison Tool.

Evaluates a trained RL-ACK DQN agent against all 5 static baseline policies
under strictly identical random seeds, network topology, and traffic traces.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.network import SmartKitchenSimulation
from policies.baselines import get_all_policies, DQNPolicy
from rl.dqn_agent import DQNAgent


def evaluate_policies(model_path: str = None, duration_s: float = 30.0,
                      num_devices: int = 10, utilization: float = 0.7,
                      ack_loss: float = 0.05, seed: int = 42):
    """
    Run evaluation across all 6 policies under identical conditions.
    """
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config['simulation']['duration_s'] = duration_s
    config['network']['default_device_count'] = num_devices
    config['network']['default_utilization'] = utilization
    config['network']['default_ack_loss'] = ack_loss

    # Load DQN agent if model exists
    agent = DQNAgent(config.get("dqn", {}), seed=seed)
    if model_path and os.path.exists(model_path):
        print(f"[EVAL] Loading DQN weights from {model_path}")
        agent.load_checkpoint(model_path)
    else:
        default_model = os.path.join(PROJECT_ROOT, "models", "dqn_checkpoint_latest.pt")
        if os.path.exists(default_model):
            print(f"[EVAL] Loading default DQN weights from {default_model}")
            agent.load_checkpoint(default_model)
        else:
            print("[EVAL] No pretrained model found, using uninitialized DQN agent.")

    policies = get_all_policies(agent)

    print("\n" + "=" * 80)
    print("EVALUATION UNDER IDENTICAL NETWORK CONDITIONS")
    print(f"Duration: {duration_s}s | Devices: {num_devices} | Util: {utilization} | Loss: {ack_loss*100:.1f}% | Seed: {seed}")
    print("=" * 80)

    rows = []
    for policy in policies:
        sim = SmartKitchenSimulation(
            config=config,
            policy=policy,
            seed=seed,
        )
        res = sim.run(progress_callback=lambda *args: None)

        mdr = res['mdr']
        lat = res['latency']
        ovh = res['overhead']
        eng = res['energy']

        rows.append({
            "Policy": policy.name,
            "MDR Overall": f"{mdr['mdr_overall']*100:.2f}%",
            "MDR (CC)": f"{mdr.get('mdr_CC', 0)*100:.2f}%",
            "Latency (ms)": f"{lat['mean_latency_ms']:.2f}",
            "P95 Lat (ms)": f"{lat['p95_latency_ms']:.2f}",
            "Overhead (KB/s)": f"{ovh['overhead_kb_per_s']:.2f}",
            "Energy (mJ/s)": f"{eng['energy_mj_per_s']:.2f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print("=" * 80 + "\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL-ACK Policies")
    parser.add_argument("--model", type=str, default="", help="Path to DQN checkpoint")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration (s)")
    parser.add_argument("--devices", type=int, default=10, help="Device count")
    parser.add_argument("--util", type=float, default=0.7, help="Network utilization target")
    parser.add_argument("--loss", type=float, default=0.05, help="ACK path loss rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    evaluate_policies(
        model_path=args.model if args.model else None,
        duration_s=args.duration,
        num_devices=args.devices,
        utilization=args.util,
        ack_loss=args.loss,
        seed=args.seed,
    )
