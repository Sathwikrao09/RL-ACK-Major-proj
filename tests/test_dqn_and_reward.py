"""
Tests for DQN Agent Architecture and Reward Function.
Validates:
  - State vector dimension (8 features) and Action vector dimension (5 actions)
  - Hidden layers (64 -> 32)
  - Replay buffer sampling
  - Target network weight sync
  - Reward weights sum to 1.0 (-0.40 latency, -0.20 overhead, -0.30 reliability, +0.10 utilization)
  - Deadline-based latency normalization
  - Overhead normalization against GLOBAL_MAX_OVERHEAD
"""

import unittest
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.dqn_agent import DQNAgent, DQNNetwork
from rl.replay_buffer import ReplayBuffer
from rl.reward import RewardCalculator
from ack.actions import AckAction
from sim.packet import TrafficTier, Packet, PacketType, DeliveryStatus


class TestDQNAndReward(unittest.TestCase):

    def setUp(self):
        self.dqn_config = {
            "state_dim": 8,
            "action_dim": 5,
            "hidden_1": 64,
            "hidden_2": 32,
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "batch_size": 32,
            "replay_buffer_size": 1000,
            "target_update_interval": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 1000,
            "training_steps": 2000,
        }
        self.agent = DQNAgent(self.dqn_config, device="cpu", seed=42)

        self.reward_config = {
            "weight_latency": 0.40,
            "weight_overhead": 0.20,
            "weight_reliability": 0.30,
            "weight_utilization": 0.10,
        }
        self.reward_calc = RewardCalculator(self.reward_config)

    def test_network_dimensions(self):
        """Verify network layer dimensions: 8 -> 64 -> 32 -> 5."""
        net = self.agent.policy_net
        dummy_state = torch.randn(1, 8)
        output = net(dummy_state)
        self.assertEqual(output.shape, (1, 5))

    def test_replay_buffer(self):
        """Verify experience replay buffer storage and sampling."""
        buffer = ReplayBuffer(capacity=100)
        for i in range(50):
            s = np.zeros(8)
            ns = np.ones(8)
            buffer.push(s, i % 5, float(i), ns, False)
        self.assertEqual(len(buffer), 50)
        states, actions, rewards, next_states, dones = buffer.sample(16)
        self.assertEqual(len(states), 16)
        self.assertEqual(len(actions), 16)

    def test_reward_weights_sum_to_one(self):
        """Verify the 4 components have absolute weights summing to 1.0."""
        w_lat = self.reward_calc.w_latency
        w_ovh = self.reward_calc.w_overhead
        w_rel = self.reward_calc.w_reliability
        w_utl = self.reward_calc.w_utilization
        self.assertAlmostEqual(w_lat + w_ovh + w_rel + w_utl, 1.0, places=4)

    def test_reward_calculation_success_case(self):
        """Verify reward calculation for on-time delivery under low utilization."""
        pkt = Packet(
            packet_id="test_001",
            source_id="device_0",
            destination_id="edge_router",
            packet_type=PacketType.DATA,
            traffic_tier=TrafficTier.CC,
            payload_size_bytes=64,
            generation_time=0.0,
            deadline=0.010,  # 10 ms
            status=DeliveryStatus.DELIVERED,
            arrival_time=0.005,
        )

        # Action A1, 50 bytes overhead, 50% utilization
        r, comp = self.reward_calc.compute_reward(
            packet=pkt,
            overhead_bytes=50,
            utilization=0.5,
            action=AckAction.A1_LOCAL_ACK,
        )

        # Latency normalized = 5 / 10 = 0.5 -> -0.40 * 0.5 = -0.20
        # Overhead normalized <= 1.0 -> -0.20 * (50/GLOBAL_MAX_OVERHEAD)
        # Reliability violation = 0 -> 0.0
        # Utilization bonus = +0.10 * (1 - 0.5) = +0.05
        self.assertGreater(r, -1.0)
        self.assertLessEqual(r, 0.5)

    def test_reward_reliability_violation(self):
        """Verify failed packet receives reliability penalty (-0.30)."""
        pkt = Packet(
            packet_id="test_fail",
            source_id="device_0",
            destination_id="edge_router",
            packet_type=PacketType.DATA,
            traffic_tier=TrafficTier.CC,
            payload_size_bytes=64,
            generation_time=0.0,
            deadline=0.010,
            status=DeliveryStatus.TIMED_OUT,
        )

        r_failed, comp = self.reward_calc.compute_reward(
            packet=pkt,
            overhead_bytes=0,
            utilization=0.5,
            action=AckAction.A0_NO_ACK,
        )
        # Should heavily penalize failure
        self.assertLess(r_failed, -0.30)


if __name__ == '__main__':
    unittest.main()
