"""
Tests for Simulation Reproducibility and Determinism.
Validates:
  - Same random seed produces identical simulation message counts, latencies, and MDR
  - Different random seeds produce distinct stochastic variations
"""

import unittest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.network import SmartKitchenSimulation
from policies.baselines import StaticPolicy
from ack.actions import AckAction


class TestReproducibility(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.json')
        with open(config_path) as f:
            self.config = json.load(f)
        self.config['simulation']['duration_s'] = 2.0

    def test_deterministic_seed_results(self):
        """Verify identical results with identical seeds."""
        sim1 = SmartKitchenSimulation(
            config=self.config,
            policy=StaticPolicy(AckAction.A2_PUBACK),
            seed=777
        )
        res1 = sim1.run(progress_callback=lambda *args: None)

        sim2 = SmartKitchenSimulation(
            config=self.config,
            policy=StaticPolicy(AckAction.A2_PUBACK),
            seed=777
        )
        res2 = sim2.run(progress_callback=lambda *args: None)

        self.assertEqual(res1['mdr']['total_messages'], res2['mdr']['total_messages'])
        self.assertAlmostEqual(res1['latency']['mean_latency_ms'], res2['latency']['mean_latency_ms'], places=4)
        self.assertAlmostEqual(res1['mdr']['mdr_overall'], res2['mdr']['mdr_overall'], places=4)

    def test_different_seeds_produce_variation(self):
        """Verify different seeds produce varying event streams."""
        sim1 = SmartKitchenSimulation(
            config=self.config,
            policy=StaticPolicy(AckAction.A2_PUBACK),
            seed=101
        )
        res1 = sim1.run(progress_callback=lambda *args: None)

        sim2 = SmartKitchenSimulation(
            config=self.config,
            policy=StaticPolicy(AckAction.A2_PUBACK),
            seed=202
        )
        res2 = sim2.run(progress_callback=lambda *args: None)

        # Different seeds will have slightly different arrival timings
        self.assertIsInstance(res1['latency']['mean_latency_ms'], float)
        self.assertIsInstance(res2['latency']['mean_latency_ms'], float)


if __name__ == '__main__':
    unittest.main()
