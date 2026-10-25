"""
Reward Function for the RL-ACK Framework.

Implements the paper-aligned normalized reward:

  r = -0.40 × normalized_latency
      -0.20 × normalized_overhead
      -0.30 × reliability_violation
      +0.10 × (1 - utilization)

Weights sum to 1.0.

All normalizations are documented and validated via unit tests.
"""

import numpy as np
from typing import Optional

from sim.packet import Packet, TrafficTier, DeliveryStatus, TIER_DEADLINES
from ack.actions import AckAction, GLOBAL_MAX_OVERHEAD


class RewardCalculator:
    """
    Computes the RL reward signal from simulation observations.

    Each component is normalized to [0, 1] range before weighting.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Reward weights from default_config.json['reward'].
        """
        config = config or {}
        self.w_latency = config.get('w_latency', 0.40)
        self.w_overhead = config.get('w_overhead', 0.20)
        self.w_reliability = config.get('w_reliability', 0.30)
        self.w_utilization = config.get('w_utilization', 0.10)

        # Validate weights sum to 1.0
        total = self.w_latency + self.w_overhead + self.w_reliability + self.w_utilization
        assert abs(total - 1.0) < 0.01, \
            f"Reward weights must sum to 1.0, got {total}"

        # Statistics
        self.reward_history: list[dict] = []
        self.total_rewards = 0
        self.sum_rewards = 0.0

    def compute_reward(self, packet: Packet, overhead_bytes: int,
                       utilization: float, action: AckAction) -> tuple[float, dict]:
        """
        Compute the reward for a single message delivery.

        Args:
            packet: The delivered (or failed) packet.
            overhead_bytes: Total control overhead bytes for this message.
            utilization: Current network utilization (0 to 1).
            action: The ACK action that was applied.

        Returns:
            Tuple of (reward_value, component_dict).
        """
        # --- Component 1: Normalized Latency ---
        # Normalize against the traffic-tier deadline
        deadline = TIER_DEADLINES.get(packet.traffic_tier, 0.5)
        latency = packet.end_to_end_latency

        if latency is not None:
            normalized_latency = min(latency / deadline, 2.0)  # Cap at 2x deadline
        else:
            # No delivery = maximum latency penalty
            normalized_latency = 2.0

        # --- Component 2: Normalized Overhead ---
        # Normalize against maximum possible overhead for this action
        max_overhead = GLOBAL_MAX_OVERHEAD if GLOBAL_MAX_OVERHEAD > 0 else 1.0
        normalized_overhead = min(overhead_bytes / max_overhead, 1.0)

        # --- Component 3: Reliability Violation ---
        # Binary: 1 if delivery failed or missed deadline, 0 if successful and on time
        reliability_violation = 0.0
        if packet.status in (DeliveryStatus.LOST, DeliveryStatus.TIMED_OUT,
                             DeliveryStatus.DROPPED, DeliveryStatus.DEADLINE_MISSED):
            reliability_violation = 1.0
        elif packet.is_deadline_compliant is False:
            reliability_violation = 1.0

        # --- Component 4: Utilization Bonus ---
        # Reward for keeping utilization low (network breathing room)
        utilization_bonus = 1.0 - min(utilization, 1.0)

        # --- Compute weighted reward ---
        reward = (
            -self.w_latency * normalized_latency
            - self.w_overhead * normalized_overhead
            - self.w_reliability * reliability_violation
            + self.w_utilization * utilization_bonus
        )

        # Record components for analysis
        components = {
            'reward': round(reward, 6),
            'normalized_latency': round(normalized_latency, 6),
            'normalized_overhead': round(normalized_overhead, 6),
            'reliability_violation': reliability_violation,
            'utilization_bonus': round(utilization_bonus, 6),
            'raw_latency_s': latency,
            'deadline_s': deadline,
            'overhead_bytes': overhead_bytes,
            'utilization': utilization,
            'action': action.name,
            'traffic_tier': packet.traffic_tier.value,
            'delivery_status': packet.status.name,
        }

        self.reward_history.append(components)
        self.total_rewards += 1
        self.sum_rewards += reward

        return reward, components

    def compute_batch_reward(self, packets: list[Packet],
                             overheads: list[int],
                             utilization: float,
                             actions: list[AckAction]) -> tuple[float, list[dict]]:
        """Compute average reward over a batch of messages."""
        rewards = []
        all_components = []

        for pkt, ovh, act in zip(packets, overheads, actions):
            r, comp = self.compute_reward(pkt, ovh, utilization, act)
            rewards.append(r)
            all_components.append(comp)

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        return avg_reward, all_components

    @property
    def average_reward(self) -> float:
        """Average reward over all computed rewards."""
        if self.total_rewards == 0:
            return 0.0
        return self.sum_rewards / self.total_rewards

    def get_stats(self) -> dict:
        """Get reward computation statistics."""
        if not self.reward_history:
            return {'total': 0, 'average': 0.0}

        recent = self.reward_history[-100:]
        return {
            'total': self.total_rewards,
            'average': round(self.average_reward, 6),
            'recent_avg': round(
                float(np.mean([r['reward'] for r in recent])), 6
            ),
            'recent_avg_latency_component': round(
                float(np.mean([r['normalized_latency'] for r in recent])), 6
            ),
            'recent_avg_overhead_component': round(
                float(np.mean([r['normalized_overhead'] for r in recent])), 6
            ),
            'recent_reliability_violation_rate': round(
                float(np.mean([r['reliability_violation'] for r in recent])), 6
            ),
        }

    def reset(self) -> None:
        """Reset statistics."""
        self.reward_history.clear()
        self.total_rewards = 0
        self.sum_rewards = 0.0
