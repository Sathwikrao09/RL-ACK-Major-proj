"""
Experiment Execution Framework for RL-ACK Smart Kitchen Simulator.

Supports:
  - Quick-demo mode (fast parameter sweep for rapid verification)
  - Full-evaluation mode (N in [10, 20, 30], utilization 0.5-0.9, loss 1%-15%, 10 seeds, 300s)
  - DQN pretraining and checkpointing
  - 6-policy comparison under identical seeds and traffic traces
  - Interrupt-safe execution with resume capability (skips already computed configs)
  - Saving raw per-run metrics and per-message logs
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.network import SmartKitchenSimulation
from policies.baselines import get_all_policies, StaticPolicy, DQNPolicy
from rl.dqn_agent import DQNAgent
from ack.actions import AckAction


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_dqn_agent(config: dict, steps_target: int = 5000, seed: int = 42) -> DQNAgent:
    """
    Pretrain or fine-tune the DQN agent using simulation interaction steps.
    """
    print(f"\n[TRAIN] Pretraining RL-ACK DQN agent (target steps={steps_target})...")
    dqn_config = config.get("dqn", {})
    agent = DQNAgent(dqn_config, seed=seed)

    # Temporary simulation for training
    train_cfg = json.loads(json.dumps(config))
    train_cfg['simulation']['duration_s'] = 60.0  # Training episodes
    train_cfg['network']['default_device_count'] = 20
    train_cfg['network']['default_utilization'] = 0.7
    train_cfg['network']['default_ack_loss'] = 0.05

    policy = DQNPolicy(agent, deterministic=False)
    sim = SmartKitchenSimulation(config=train_cfg, policy=policy, dqn_agent=agent, seed=seed)

    while agent.train_count < (steps_target // 20) and agent.step_count < steps_target:
        sim.run(progress_callback=lambda *args: None)
        if agent.step_count >= steps_target:
            break

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, "dqn_checkpoint_latest.pt")
    agent.save_checkpoint(checkpoint_path)
    print(f"[TRAIN] Finished training: {agent.step_count} interaction steps, "
          f"{agent.train_count} training updates. Model saved to {checkpoint_path}")
    return agent


def run_experiment_grid(mode: str = "quick", resume: bool = True):
    """
    Run the parameter sweep comparing all 6 policies.
    """
    config = load_config()
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    manifest_path = os.path.join(RESULTS_DIR, "experiment_manifest.json")

    # Define grid based on mode
    if mode == "quick":
        device_counts = [10, 20]
        utilizations = [0.6, 0.8]
        ack_loss_rates = [0.05, 0.10]
        seeds = [42, 101]
        duration_s = 10.0
        train_steps = 1000
    else:  # full
        device_counts = [10, 20, 30]
        utilizations = [0.5, 0.6, 0.7, 0.8, 0.9]
        ack_loss_rates = [0.01, 0.05, 0.10, 0.15]
        seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
        duration_s = 300.0
        train_steps = 10000

    print("=" * 70)
    print(f"RL-ACK EXPERIMENT RUNNER — Mode: {mode.upper()}")
    print(f"Devices:       {device_counts}")
    print(f"Utilizations:  {utilizations}")
    print(f"ACK Loss:      {[f'{l*100:.0f}%' for l in ack_loss_rates]}")
    print(f"Seeds:         {len(seeds)} seeds")
    print(f"Duration:      {duration_s}s per run")
    print("=" * 70)

    # Train or load DQN agent
    checkpoint_path = os.path.join(MODELS_DIR, "dqn_checkpoint_latest.pt")
    agent = DQNAgent(config.get("dqn", {}), seed=seeds[0])
    if os.path.exists(checkpoint_path):
        print(f"[INIT] Loading existing DQN checkpoint: {checkpoint_path}")
        agent.load_checkpoint(checkpoint_path)
    else:
        agent = train_dqn_agent(config, steps_target=train_steps, seed=seeds[0])

    # All six policies to evaluate
    policies = get_all_policies(agent)

    completed_runs = 0
    skipped_runs = 0
    total_runs = (len(device_counts) * len(utilizations) *
                  len(ack_loss_rates) * len(seeds) * len(policies))

    start_wall_time = time.time()

    for n_dev in device_counts:
        for util in utilizations:
            for loss in ack_loss_rates:
                for seed in seeds:
                    for policy in policies:
                        run_id = f"N{n_dev}_u{int(util*100)}_loss{int(loss*100)}_s{seed}_{policy.name.lower().replace(' ', '_')}"
                        result_file = os.path.join(RAW_RESULTS_DIR, f"{run_id}.json")

                        if resume and os.path.exists(result_file):
                            skipped_runs += 1
                            continue

                        # Prepare run config
                        run_cfg = json.loads(json.dumps(config))
                        run_cfg['simulation']['duration_s'] = duration_s
                        run_cfg['network']['default_device_count'] = n_dev
                        run_cfg['network']['default_utilization'] = util
                        run_cfg['network']['default_ack_loss'] = loss

                        # Execute simulation run
                        sim = SmartKitchenSimulation(
                            config=run_cfg,
                            policy=policy,
                            dqn_agent=agent if isinstance(policy, DQNPolicy) else None,
                            seed=seed,
                        )

                        metrics = sim.run(progress_callback=lambda *args: None)
                        metrics['run_id'] = run_id
                        metrics['timestamp'] = time.time()

                        with open(result_file, "w", encoding="utf-8") as rf:
                            json.dump(metrics, rf, indent=2)

                        completed_runs += 1
                        pct = (completed_runs + skipped_runs) / total_runs * 100
                        print(f"[{pct:5.1f}%] Completed {run_id} | "
                              f"MDR={metrics['mdr']['mdr_overall']:.4f}, "
                              f"Latency={metrics['latency']['mean_latency_ms']:.2f}ms")

    elapsed = time.time() - start_wall_time
    print("\n" + "=" * 70)
    print(f"[SUMMARY] Experiment Execution Complete in {elapsed:.1f}s")
    print(f"Total runs:     {total_runs}")
    print(f"Newly executed: {completed_runs}")
    print(f"Skipped:        {skipped_runs}")
    print(f"Raw data saved: {RAW_RESULTS_DIR}")
    print("=" * 70)

    # Save manifest
    manifest = {
        "mode": mode,
        "completed_runs": completed_runs,
        "skipped_runs": skipped_runs,
        "total_runs": total_runs,
        "device_counts": device_counts,
        "utilizations": utilizations,
        "ack_loss_rates": ack_loss_rates,
        "seeds": seeds,
        "duration_s": duration_s,
        "policies": [p.name for p in policies],
        "timestamp": time.time(),
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-ACK Experiment Runner")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="Execution mode: quick (default) or full")
    parser.add_argument("--no-resume", action="store_true",
                        help="Overwrite existing results without resuming")
    args = parser.parse_args()

    run_experiment_grid(mode=args.mode, resume=not args.no_resume)
