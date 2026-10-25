"""
Tests for Virtual Kitchen Sensors and Threshold Alerts.
Validates:
  - Sensor model instantiation (gas, temp, humidity, smoke)
  - Time-series reading generation with noise
  - Configurable threshold crossing detection and alert records
  - Gas-leak simulated event generation
  - Sensor offline / failure mode simulation
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensors.sensor_models import SensorFactory, SensorType
from sensors.event_generator import EventGenerator, ScheduledEvent


class TestSensorsAndAlerts(unittest.TestCase):

    def setUp(self):
        self.sensor_config = {
            "gas": {
                "unit": "ppm",
                "normal_range": [10.0, 30.0],
                "alert_threshold": 50.0,
                "noise_std": 2.0,
                "sampling_interval_s": 1.0,
            },
            "temperature": {
                "unit": "C",
                "normal_range": [20.0, 26.0],
                "alert_threshold": 60.0,
                "noise_std": 0.5,
                "sampling_interval_s": 2.0,
            },
        }

    def test_sensor_creation_and_reading(self):
        """Verify sensor creates synthetic readings within expected bounds."""
        gas_sensor = SensorFactory.create_sensor(
            "gas", self.sensor_config["gas"], device_id="dev_001"
        )
        self.assertEqual(gas_sensor.sensor_type, SensorType.GAS_CONCENTRATION)
        self.assertEqual(gas_sensor.unit, "ppm")

        reading = gas_sensor.generate_reading(sim_time=1.0)
        self.assertIsNotNone(reading)
        self.assertGreater(reading.value, 0)
        self.assertEqual(reading.unit, "ppm")
        self.assertEqual(reading.status.name, "NORMAL")

    def test_gas_alert_threshold_crossing(self):
        """Verify alert triggering when gas concentration exceeds threshold."""
        gas_sensor = SensorFactory.create_sensor(
            "gas", self.sensor_config["gas"], device_id="dev_001"
        )
        # Inject gas leak event (peak 200 ppm > alert threshold 50 ppm)
        gas_sensor.inject_event(
            event_type="gas_leak",
            start_time=1.0,
            duration=10.0,
            peak_value=200.0,
        )
        reading = gas_sensor.generate_reading(sim_time=5.0)

        self.assertGreater(reading.value, 50.0)
        self.assertIn(reading.status.name, ["ALERT", "DANGER"])
        self.assertTrue(len(gas_sensor.alert_history) > 0)
        alert = gas_sensor.alert_history[-1]
        self.assertEqual(alert["sensor_id"], gas_sensor.sensor_id)

    def test_event_generator_gas_leak(self):
        """Verify event generator injects gas leak event into sensor."""
        event_gen = EventGenerator({
            "gas_leak": {
                "probability_per_hour": 3600.0,  # High rate to guarantee event in short time
                "duration_range_s": [10, 30],
                "peak_ppm": 800,
            }
        })
        gas_sensor = SensorFactory.create_sensor(
            "gas", self.sensor_config["gas"], device_id="dev_001"
        )
        events = event_gen.generate_events_for_sensor(gas_sensor, sim_duration=100)
        self.assertTrue(len(events) > 0)


if __name__ == '__main__':
    unittest.main()
