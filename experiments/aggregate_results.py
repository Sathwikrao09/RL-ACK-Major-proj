"""
Results Aggregator for RL-ACK Experiments.

Aggregates raw experiment JSON files across seeds:
  - Computes mean and standard deviation for each metric
  - Generates summary CSV files for easy dashboard loading and paper comparison
  - Exports per-tier deadline compliance and latency stats
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
AGG_RESULTS_DIR = os.path.join(RESULTS_DIR, "aggregated")


def aggregate_all_results():
    os.makedirs(AGG_RESULTS_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(RAW_RESULTS_DIR, "*.json"))
    if not json_files:
        print("[AGGREGATOR] No raw result files found in", RAW_RESULTS_DIR)
        return

    print(f"[AGGREGATOR] Processing {len(json_files)} raw result files...")

    records = []
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = data.get("config", {})
            lat = data.get("latency", {})
            ovh = data.get("overhead", {})
            mdr = data.get("mdr", {})
            eng = data.get("energy", {})
            que = data.get("queue", {})
            utl = data.get("utilization", {})

            rec = {
                "policy": cfg.get("policy", "Unknown"),
                "num_devices": cfg.get("num_devices", 10),
                "utilization_target": cfg.get("utilization_target", 0.7),
                "ack_loss_rate": cfg.get("ack_loss_rate", 0.05),
                "seed": cfg.get("seed", 42),
                # Latency
                "mean_latency_ms": lat.get("mean_latency_ms", 0.0),
                "median_latency_ms": lat.get("median_latency_ms", 0.0),
                "p95_latency_ms": lat.get("p95_latency_ms", 0.0),
                "jitter_ms": lat.get("jitter_ms", 0.0),
                # Overhead
                "overhead_kb_s": ovh.get("overhead_kb_per_s", 0.0),
                "control_overhead_pct": ovh.get("control_plane_overhead_pct", 0.0),
                # Delivery
                "mdr_overall": mdr.get("mdr_overall", 0.0),
                "delivery_rate": mdr.get("delivery_rate", 0.0),
                "mdr_CC": mdr.get("mdr_CC", 0.0),
                "mean_latency_CC_ms": mdr.get("mean_latency_CC_ms", 0.0),
                "mdr_HBS": mdr.get("mdr_HBS", 0.0),
                "mean_latency_HBS_ms": mdr.get("mean_latency_HBS_ms", 0.0),
                "mdr_BET": mdr.get("mdr_BET", 0.0),
                "mean_latency_BET_ms": mdr.get("mean_latency_BET_ms", 0.0),
                # Energy & Queue
                "energy_mj_s": eng.get("energy_mj_per_s", 0.0),
                "mean_queue_occupancy": que.get("mean_queue_occupancy", 0.0),
                "actual_utilization": utl.get("mean_utilization", 0.0),
            }
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Error reading {fpath}: {e}")

    if not records:
        print("[AGGREGATOR] No valid records extracted.")
        return

    df = pd.DataFrame(records)

    # Save per-run dataframe
    raw_csv = os.path.join(AGG_RESULTS_DIR, "all_runs.csv")
    df.to_csv(raw_csv, index=False)
    print(f"[AGGREGATOR] Saved all runs to {raw_csv}")

    # Group and aggregate
    group_cols = ["policy", "num_devices", "utilization_target", "ack_loss_rate"]
    metric_cols = [c for c in df.columns if c not in group_cols and c != "seed"]

    agg_df = df.groupby(group_cols)[metric_cols].agg(['mean', 'std']).reset_index()

    # Flatten multi-level columns
    agg_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg_df.columns
    ]

    summary_csv = os.path.join(AGG_RESULTS_DIR, "summary_metrics.csv")
    agg_df.to_csv(summary_csv, index=False)
    print(f"[AGGREGATOR] Saved aggregated summary to {summary_csv}")

    # Save policy comparison table at default settings (N=10, util=0.7, loss=0.05)
    default_slice = df[
        (df["num_devices"] == 10) &
        (np.isclose(df["utilization_target"], 0.7, atol=0.05)) &
        (np.isclose(df["ack_loss_rate"], 0.05, atol=0.02))
    ]
    if not default_slice.empty:
        policy_comp = default_slice.groupby("policy")[[
            "mean_latency_ms", "overhead_kb_s", "mdr_overall", "mdr_CC", "energy_mj_s"
        ]].mean().reset_index()
        comp_csv = os.path.join(AGG_RESULTS_DIR, "policy_comparison_baseline.csv")
        policy_comp.to_csv(comp_csv, index=False)
        print(f"[AGGREGATOR] Saved baseline policy comparison to {comp_csv}")

    print("[AGGREGATOR] Aggregation finished successfully.")


if __name__ == "__main__":
    aggregate_all_results()
