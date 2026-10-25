"""
Tests for Traffic Generation and Classification.
Validates:
  - Traffic tier classification (CC, HBS, BET)
  - Deadline assignments (10ms, 50ms, 500ms)
  - Traffic mix validation (must sum to 100%)
  - Device count allocation per tier
  - Rate scaling based on utilization
"""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from traffic.generator import TrafficGenerator, TrafficConfig
from sim.packet import TrafficTier, TIER_DEADLINES


class TestTrafficGenerator(unittest.TestCase):

    def setUp(self):
        self.config = {
            "tiers": {
                "critical_control": {
                    "deadline_ms": 10,
                    "target_share": 0.20,
                    "arrival_rate_range": [1, 5],
                    "payload_size_bytes": 64,
                },
                "high_bandwidth_streaming": {
                    "deadline_ms": 50,
                    "target_share": 0.30,
                    "arrival_rate_range": [15, 30],
                    "payload_size_bytes": 1024,
                },
                "best_effort_telemetry": {
                    "deadline_ms": 500,
                    "target_share": 0.50,
                    "arrival_rate_range": [0.1, 2],
                    "payload_size_bytes": 128,
                },
            }
        }
        self.generator = TrafficGenerator(self.config)

    def test_tier_deadlines(self):
        """Verify the paper's specified deadlines per tier."""
        self.assertEqual(TIER_DEADLINES[TrafficTier.CC], 0.010)   # 10 ms
        self.assertEqual(TIER_DEADLINES[TrafficTier.HBS], 0.050)  # 50 ms
        self.assertEqual(TIER_DEADLINES[TrafficTier.BET], 0.500)  # 500 ms

    def test_traffic_composition_sum(self):
        """Verify traffic mix composition sums to 100%."""
        cfg = TrafficConfig(self.config)
        total_share = sum(cfg.composition.values())
        self.assertAlmostEqual(total_share, 1.0, places=4)

    def test_device_counts_allocation(self):
        """Verify device allocation for N=10, 20, 30 matches tier shares."""
        cfg = TrafficConfig(self.config)
        for total_devs in [10, 20, 30]:
            dev_counts = cfg.get_device_counts(total_devs)
            self.assertEqual(sum(dev_counts.values()), total_devs)
            # CC should be ~20%
            self.assertGreaterEqual(dev_counts[TrafficTier.CC], 1)
            # BET should be ~50%
            self.assertGreaterEqual(dev_counts[TrafficTier.BET], int(0.4 * total_devs))

    def test_message_classification(self):
        """Verify packet classification by topic and metadata."""
        tier_gas = self.generator.classify_message("safety/gas_alarm", 64)
        self.assertEqual(tier_gas, TrafficTier.CC)

        tier_cam = self.generator.classify_message("camera/stream", 1024)
        self.assertEqual(tier_cam, TrafficTier.HBS)

        tier_temp = self.generator.classify_message("sensor/temperature", 128)
        self.assertEqual(tier_temp, TrafficTier.BET)


if __name__ == '__main__':
    unittest.main()
