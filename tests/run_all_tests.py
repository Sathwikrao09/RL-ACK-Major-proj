"""
Master test runner for RL-ACK Smart Kitchen IoT Simulator.
Executes all unit and integration tests and outputs a structured report.
"""

import unittest
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        'test_traffic',
        'test_actions',
        'test_dqn_and_reward',
        'test_sensors_and_alerts',
        'test_simulation_and_metrics',
        'test_reproducibility',
    ]

    for mod_name in test_modules:
        mod = __import__(mod_name)
        suite.addTests(loader.loadTestsFromModule(mod))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_suite()
    sys.exit(0 if success else 1)
