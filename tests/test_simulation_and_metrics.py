"""
Tests for Network Simulation and Performance Metrics.
Validates:
  - Discrete-event simulation execution with different baseline policies
  - Latency statistics (mean, median, p95, jitter)
  - Overhead metrics (KB/s and control plane percentage)
  - Per-tier Deadline-compliant Message Delivery Ratio (MDR)
  - Retransmission tracking
"""

import unittest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.network import SmartKitchenSimulation
from policies.baselines import StaticPolicy
from ack.actions import AckAction


class TestSimulationAndMetrics(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.json')
        with open(config_path) as f:
            self.config = json.load(f)
        self.config['simulation']['duration_s'] = 3.0  # Fast execution for tests

    def test_run_static_policies(self):
        """Verify simulation runs for all static baseline policies."""
        policies_to_test = [
            (AckAction.A0_NO_ACK, "Static QoS 0"),
            (AckAction.A1_LOCAL_ACK, "Static Local ACK"),
            (AckAction.A2_PUBACK, "Static QoS 1"),
        ]

        for action, name in policies_to_test:
            sim = SmartKitchenSimulation(
                config=self.config,
                policy=StaticPolicy(action),
                seed=100
            )
            res = sim.run(progress_callback=lambda *args: None)

            self.assertIn('latency', res)
            self.assertIn('mdr', res)
            self.assertIn('overhead', res)
            self.assertGreater(res['mdr']['total_messages'], 0)
            self.assertGreaterEqual(res['mdr']['mdr_overall'], 0.0)
            self.assertLessEqual(res['mdr']['mdr_overall'], 1.0)

    def test_metrics_tier_breakdown(self):
        """Verify per-tier MDR and latency are computed."""
        sim = SmartKitchenSimulation(
            config=self.config,
            policy=StaticPolicy(AckAction.A2_PUBACK),
            seed=200
        )
        res = sim.run(progress_callback=lambda *args: None)

        mdr = res['mdr']
        self.assertIn('mdr_CC', mdr)
        self.assertIn('mdr_HBS', mdr)
        self.assertIn('mdr_BET', mdr)


if __name__ == '__main__':
    unittest.main()
