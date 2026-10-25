"""
Performance Metrics Collector for the RL-ACK Simulator.

Collects and aggregates all metrics from simulation events:
  - Latency (mean, median, p95, jitter)
  - Overhead (KB/s, control-plane)
  - Deadline-compliant MDR (overall and per-tier)
  - Loss rates (forward, ACK-path)
  - Retransmission counts
  - Energy consumption (modeled, mJ/s)
  - Queue occupancy
  - Utilization (actual measured)
  - DQN reward and training loss
  - ACK action distribution by tier

Each metric is computed from actual simulation packet events,
not from analytical formulas.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

from sim.packet import Packet, TrafficTier, DeliveryStatus


class MetricsCollector:
    """
    Collects per-message records and computes aggregate metrics.

    All metrics are derived from actual packet event data.
    """

    def __init__(self):
        self.message_records: list[dict] = []
        self.periodic_samples: list[dict] = []
        self.dqn_rewards: list[float] = []
        self.dqn_losses: list[float] = []
        self.action_counts = defaultdict(lambda: defaultdict(int))  # tier -> action -> count
        self.energy_samples: list[tuple[float, float]] = []  # (time, mJ/s)
        self.queue_samples: list[tuple[float, float]] = []   # (time, occupancy)
        self.utilization_samples: list[tuple[float, float]] = []  # (time, utilization)

    def record_message(self, packet: Packet, overhead_bytes: int = 0,
                       energy_mj: float = 0.0) -> None:
        """Record a completed message (delivered or failed)."""
        record = packet.to_dict()
        record['overhead_bytes'] = overhead_bytes
        record['energy_mj'] = energy_mj
        self.message_records.append(record)

        # Track action distribution by tier
        if packet.ack_action >= 0:
            self.action_counts[packet.traffic_tier.value][packet.ack_action] += 1

    def record_dqn_step(self, reward: float, loss: Optional[float] = None) -> None:
        """Record a DQN training step."""
        self.dqn_rewards.append(reward)
        if loss is not None:
            self.dqn_losses.append(loss)

    def record_periodic(self, sim_time: float, utilization: float,
                        queue_occupancy: float, energy_rate_mj_s: float) -> None:
        """Record periodic monitoring sample."""
        self.periodic_samples.append({
            'time': sim_time,
            'utilization': utilization,
            'queue_occupancy': queue_occupancy,
            'energy_rate': energy_rate_mj_s,
        })
        self.utilization_samples.append((sim_time, utilization))
        self.queue_samples.append((sim_time, queue_occupancy))
        self.energy_samples.append((sim_time, energy_rate_mj_s))

    def compute_metrics(self) -> dict:
        """
        Compute all aggregate metrics from recorded data.

        Returns a comprehensive metrics dictionary.
        """
        if not self.message_records:
            return self._empty_metrics()

        df = pd.DataFrame(self.message_records)

        # Separate delivered and failed messages
        delivered = df[df['status'].isin(['DELIVERED', 'ACKED'])]
        failed = df[df['status'].isin(['LOST', 'TIMED_OUT', 'DROPPED', 'DEADLINE_MISSED'])]

        # --- Latency Metrics ---
        latencies = delivered['latency'].dropna()
        latency_metrics = {
            'mean_latency_ms': float(latencies.mean() * 1000) if len(latencies) > 0 else 0.0,
            'median_latency_ms': float(latencies.median() * 1000) if len(latencies) > 0 else 0.0,
            'p95_latency_ms': float(latencies.quantile(0.95) * 1000) if len(latencies) > 0 else 0.0,
            'jitter_ms': float(latencies.std() * 1000) if len(latencies) > 1 else 0.0,
            'min_latency_ms': float(latencies.min() * 1000) if len(latencies) > 0 else 0.0,
            'max_latency_ms': float(latencies.max() * 1000) if len(latencies) > 0 else 0.0,
        }

        # --- Overhead Metrics ---
        total_overhead_bytes = int(df['overhead_bytes'].sum())
        sim_duration = df['generation_time'].max() - df['generation_time'].min()
        sim_duration = max(sim_duration, 1.0)

        overhead_metrics = {
            'total_overhead_bytes': total_overhead_bytes,
            'overhead_kb_per_s': round(total_overhead_bytes / 1024 / sim_duration, 4),
            'control_plane_overhead_pct': round(
                total_overhead_bytes / max(1, int(df['total_size'].sum()) + total_overhead_bytes) * 100, 2
            ),
        }

        # --- Deadline Compliance (MDR) ---
        total_messages = len(df)
        deadline_compliant = df['deadline_compliant'].sum() if 'deadline_compliant' in df.columns else 0
        # Count True values
        if 'deadline_compliant' in df.columns:
            compliant_count = int(df['deadline_compliant'].fillna(False).sum())
        else:
            compliant_count = 0

        mdr_metrics = {
            'total_messages': total_messages,
            'delivered_count': len(delivered),
            'failed_count': len(failed),
            'deadline_compliant_count': compliant_count,
            'mdr_overall': round(compliant_count / max(1, total_messages), 4),
            'delivery_rate': round(len(delivered) / max(1, total_messages), 4),
        }

        # Per-tier MDR
        for tier in TrafficTier:
            tier_df = df[df['traffic_tier'] == tier.value]
            if len(tier_df) > 0:
                tier_compliant = int(tier_df['deadline_compliant'].fillna(False).sum())
                mdr_metrics[f'mdr_{tier.value}'] = round(
                    tier_compliant / len(tier_df), 4
                )
                tier_delivered = tier_df[tier_df['status'].isin(['DELIVERED', 'ACKED'])]
                tier_latencies = tier_delivered['latency'].dropna()
                mdr_metrics[f'mean_latency_{tier.value}_ms'] = \
                    float(tier_latencies.mean() * 1000) if len(tier_latencies) > 0 else 0.0
            else:
                mdr_metrics[f'mdr_{tier.value}'] = 0.0
                mdr_metrics[f'mean_latency_{tier.value}_ms'] = 0.0

        # --- Loss Metrics ---
        loss_metrics = {
            'forward_loss_count': len(df[df['status'] == 'LOST']),
            'total_retransmissions': int(df['retransmit_count'].sum()),
        }

        # --- Energy Metrics (modeled) ---
        total_energy = float(df['energy_mj'].sum())
        energy_metrics = {
            'total_energy_mj': round(total_energy, 4),
            'energy_mj_per_s': round(total_energy / sim_duration, 4),
            'label': 'modeled_energy (not measured hardware)',
        }

        # Per-tier energy
        for tier in TrafficTier:
            tier_energy = float(df[df['traffic_tier'] == tier.value]['energy_mj'].sum())
            energy_metrics[f'energy_{tier.value}_mj'] = round(tier_energy, 4)

        # --- Queue Metrics ---
        queue_metrics = {}
        if self.queue_samples:
            occ = [s[1] for s in self.queue_samples]
            queue_metrics = {
                'mean_queue_occupancy': round(float(np.mean(occ)), 4),
                'max_queue_occupancy': round(float(max(occ)), 4),
                'p95_queue_occupancy': round(float(np.percentile(occ, 95)), 4),
            }

        # --- Utilization Metrics ---
        util_metrics = {}
        if self.utilization_samples:
            utils = [s[1] for s in self.utilization_samples]
            util_metrics = {
                'mean_utilization': round(float(np.mean(utils)), 4),
                'max_utilization': round(float(max(utils)), 4),
            }

        # --- DQN Metrics ---
        dqn_metrics = {}
        if self.dqn_rewards:
            dqn_metrics = {
                'dqn_mean_reward': round(float(np.mean(self.dqn_rewards)), 6),
                'dqn_total_steps': len(self.dqn_rewards),
                'dqn_mean_loss': round(float(np.mean(self.dqn_losses)), 6) if self.dqn_losses else 0.0,
            }

        # --- Action Distribution ---
        action_dist = {}
        for tier, actions in self.action_counts.items():
            action_dist[tier] = dict(actions)

        return {
            'latency': latency_metrics,
            'overhead': overhead_metrics,
            'mdr': mdr_metrics,
            'loss': loss_metrics,
            'energy': energy_metrics,
            'queue': queue_metrics,
            'utilization': util_metrics,
            'dqn': dqn_metrics,
            'action_distribution': action_dist,
            'sim_duration_s': sim_duration,
        }

    def _empty_metrics(self) -> dict:
        """Return empty metrics structure."""
        return {
            'latency': {}, 'overhead': {}, 'mdr': {},
            'loss': {}, 'energy': {}, 'queue': {},
            'utilization': {}, 'dqn': {}, 'action_distribution': {},
            'sim_duration_s': 0,
        }

    def export_csv(self, filepath: str) -> None:
        """Export per-message records to CSV."""
        if self.message_records:
            df = pd.DataFrame(self.message_records)
            df.to_csv(filepath, index=False)

    def export_metrics_csv(self, filepath: str) -> None:
        """Export aggregate metrics to CSV."""
        metrics = self.compute_metrics()
        # Flatten nested dict
        flat = {}
        for category, values in metrics.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    flat[f"{category}.{key}"] = val
            else:
                flat[category] = values
        df = pd.DataFrame([flat])
        df.to_csv(filepath, index=False)

    def reset(self) -> None:
        """Reset all collected data."""
        self.message_records.clear()
        self.periodic_samples.clear()
        self.dqn_rewards.clear()
        self.dqn_losses.clear()
        self.action_counts.clear()
        self.energy_samples.clear()
        self.queue_samples.clear()
        self.utilization_samples.clear()
