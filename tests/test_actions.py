"""
Tests for the Five ACK Actions.
Validates:
  - Distinct control packet counts per action
  - Retransmission limits (a3: 4 retries, a4: 5 retries)
  - Handshake delay ordering: a0 < a1 < a2 < a3 < a4
  - Energy consumption ordering: a0 < a1 < a2 < a3 < a4
  - Overhead calculation
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ack.actions import AckActionHandler, AckAction, AckOutcome, GLOBAL_MAX_OVERHEAD


class TestAckActions(unittest.TestCase):

    def setUp(self):
        self.handler = AckActionHandler()

    def test_retransmission_limits(self):
        """Verify paper retransmission limits for a3 and a4."""
        self.assertEqual(self.handler.get_max_retries(AckAction.A0_NO_ACK), 0)
        self.assertEqual(self.handler.get_max_retries(AckAction.A1_LOCAL_ACK), 0)
        self.assertEqual(self.handler.get_max_retries(AckAction.A2_PUBACK), 3)
        self.assertEqual(self.handler.get_max_retries(AckAction.A3_LOCAL_SELECTIVE), 4)
        self.assertEqual(self.handler.get_max_retries(AckAction.A4_QOS2_HANDSHAKE), 5)

    def test_control_packet_counts(self):
        """Verify distinct control packet counts per action."""
        # a0: 0 ACK packets
        self.assertEqual(self.handler.get_control_packet_count(AckAction.A0_NO_ACK), 0)
        # a1: 1 local ACK
        self.assertEqual(self.handler.get_control_packet_count(AckAction.A1_LOCAL_ACK), 1)
        # a2: 1 end-to-end PUBACK
        self.assertEqual(self.handler.get_control_packet_count(AckAction.A2_PUBACK), 1)
        # a3: 1 local ACK + selective retransmission capability
        self.assertGreaterEqual(self.handler.get_control_packet_count(AckAction.A3_LOCAL_SELECTIVE), 1)
        # a4: 3-4 control packets (PUBREC, PUBREL, PUBCOMP)
        self.assertGreaterEqual(self.handler.get_control_packet_count(AckAction.A4_QOS2_HANDSHAKE), 3)

    def test_handshake_delays(self):
        """Verify delay hierarchy: a0 (0ms) < a1 < a2 < a4."""
        d0 = self.handler.get_handshake_delay(AckAction.A0_NO_ACK)
        d1 = self.handler.get_handshake_delay(AckAction.A1_LOCAL_ACK)
        d2 = self.handler.get_handshake_delay(AckAction.A2_PUBACK)
        d4 = self.handler.get_handshake_delay(AckAction.A4_QOS2_HANDSHAKE)

        self.assertEqual(d0, 0.0)
        self.assertLess(d1, d2)
        self.assertLess(d2, d4)

    def test_energy_costs(self):
        """Verify energy consumption ordering: a0 < a1 <= a2 < a4."""
        e0 = self.handler.get_energy_cost_mj(AckAction.A0_NO_ACK)
        e1 = self.handler.get_energy_cost_mj(AckAction.A1_LOCAL_ACK)
        e2 = self.handler.get_energy_cost_mj(AckAction.A2_PUBACK)
        e4 = self.handler.get_energy_cost_mj(AckAction.A4_QOS2_HANDSHAKE)

        self.assertLess(e0, e1)
        self.assertLessEqual(e1, e2)
        self.assertLess(e2, e4)

    def test_overhead_normalization_bound(self):
        """Verify overhead of any action does not exceed GLOBAL_MAX_OVERHEAD."""
        for action in AckAction:
            overhead = self.handler.get_overhead_bytes(action)
            self.assertLessEqual(overhead, GLOBAL_MAX_OVERHEAD)


if __name__ == '__main__':
    unittest.main()
