"""
Smart Kitchen Virtual Sensor Models.

Generates realistic, configurable synthetic time-series readings for:
- Gas concentration (ppm)
- Temperature (°C)
- Humidity (%)
- Smoke (% obscuration/ft)
- Pressure (hPa)
- Water leak (binary)
- Appliance status (watts)

All readings are SYNTHETIC SIMULATED VALUES — not real hardware measurements.
Each sensor has unique ID, sampling interval, noise model, calibration, and
configurable thresholds for alert generation.
"""

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class SensorType(Enum):
    GAS_CONCENTRATION = "gas_concentration"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    SMOKE = "smoke"
    PRESSURE = "pressure"
    WATER_LEAK = "water_leak"
    APPLIANCE_STATUS = "appliance_status"


class SensorStatus(Enum):
    NORMAL = auto()
    ALERT = auto()
    DANGER = auto()
    OFFLINE = auto()
    FAILED = auto()


class AlertLevel(Enum):
    NONE = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class SensorReading:
    """A single sensor reading with full metadata."""
    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: float          # Simulation time in seconds
    status: SensorStatus = SensorStatus.NORMAL
    alert_level: AlertLevel = AlertLevel.NONE
    alert_message: str = ""
    threshold: float = 0.0
    is_valid: bool = True     # False for failed/missing readings
    device_id: str = ""       # Parent IoT device ID
    sequence_num: int = 0

    def to_dict(self) -> dict:
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'value': round(self.value, 4),
            'unit': self.unit,
            'timestamp': round(self.timestamp, 4),
            'status': self.status.name,
            'alert_level': self.alert_level.name,
            'alert_message': self.alert_message,
            'threshold': self.threshold,
            'is_valid': self.is_valid,
            'device_id': self.device_id,
            'sequence_num': self.sequence_num,
        }


class VirtualSensor:
    """
    A configurable virtual sensor that produces synthetic time-series readings.

    Supports:
    - Gaussian noise around a baseline
    - Gradual drift
    - Event injection (gas leak, temperature spike, etc.)
    - Sensor failure simulation
    - Per-sensor calibration
    """

    def __init__(self, sensor_type: SensorType, config: dict,
                 device_id: str = "", rng: random.Random = None):
        """
        Args:
            sensor_type: Type of sensor.
            config: Sensor configuration dict from default_config.json.
            device_id: ID of the parent IoT device.
            rng: Random number generator for reproducibility.
        """
        self.sensor_id = f"sensor_{sensor_type.value}_{uuid.uuid4().hex[:6]}"
        self.sensor_type = sensor_type
        self.device_id = device_id
        self.rng = rng or random.Random()

        # Config
        self.unit = config.get('unit', '')
        self.normal_range = config.get('normal_range', [0, 100])
        self.alert_threshold = config.get('alert_threshold', 80)
        self.danger_threshold = config.get('danger_threshold', 95)
        self.sampling_interval_s = config.get('sampling_interval_s', 1.0)
        self.noise_std = config.get('noise_std', 1.0)
        self.calibration_factor = config.get('calibration_factor', 1.0)
        self.drift_rate = config.get('drift_rate', 0.0)

        # State
        self.baseline = (self.normal_range[0] + self.normal_range[1]) / 2.0
        self.current_value = self.baseline
        self.status = SensorStatus.NORMAL
        self.is_failed = False
        self.sequence_counter = 0
        self.last_reading_time = 0.0

        # Event state
        self.event_active = False
        self.event_type = ""
        self.event_target_value = 0.0
        self.event_ramp_rate = 0.0
        self.event_start_time = 0.0
        self.event_end_time = 0.0

        # History
        self.reading_history: list[SensorReading] = []
        self.alert_history: list[dict] = []

    def generate_reading(self, sim_time: float) -> SensorReading:
        """
        Generate a new sensor reading at the given simulation time.

        Returns:
            SensorReading with current value, status, and alerts.
        """
        self.sequence_counter += 1
        self.last_reading_time = sim_time

        # Check for sensor failure
        if self.is_failed:
            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                value=0.0,
                unit=self.unit,
                timestamp=sim_time,
                status=SensorStatus.FAILED,
                alert_level=AlertLevel.WARNING,
                alert_message=f"Sensor {self.sensor_id} is offline/failed",
                is_valid=False,
                device_id=self.device_id,
                sequence_num=self.sequence_counter,
            )
            self.reading_history.append(reading)
            return reading

        # Compute value
        value = self._compute_value(sim_time)
        self.current_value = value

        # Apply calibration
        calibrated_value = value * self.calibration_factor

        # Determine status and alerts
        status, alert_level, alert_msg = self._evaluate_thresholds(calibrated_value)
        self.status = status

        # Record alert if applicable
        if alert_level != AlertLevel.NONE:
            self.alert_history.append({
                'time': sim_time,
                'value': calibrated_value,
                'alert_level': alert_level.name,
                'message': alert_msg,
                'sensor_id': self.sensor_id,
            })

        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            value=calibrated_value,
            unit=self.unit,
            timestamp=sim_time,
            status=status,
            alert_level=alert_level,
            alert_message=alert_msg,
            threshold=self.alert_threshold,
            is_valid=True,
            device_id=self.device_id,
            sequence_num=self.sequence_counter,
        )
        self.reading_history.append(reading)
        return reading

    def _compute_value(self, sim_time: float) -> float:
        """Compute the raw sensor value at a given time."""
        # Base value with drift
        base = self.baseline + self.drift_rate * sim_time

        # Event contribution
        event_contribution = 0.0
        if self.event_active and self.event_start_time <= sim_time <= self.event_end_time:
            elapsed = sim_time - self.event_start_time
            duration = self.event_end_time - self.event_start_time
            mid_point = duration / 2

            if elapsed < mid_point:
                # Ramp up
                progress = elapsed / mid_point if mid_point > 0 else 1.0
                event_contribution = (self.event_target_value - self.baseline) * progress
            else:
                # Ramp down
                progress = (elapsed - mid_point) / mid_point if mid_point > 0 else 1.0
                event_contribution = (self.event_target_value - self.baseline) * (1.0 - progress)

        elif self.event_active and sim_time > self.event_end_time:
            self.event_active = False

        value = base + event_contribution

        # Add noise (Gaussian)
        if self.noise_std > 0:
            noise = self.rng.gauss(0, self.noise_std)
            value += noise

        # Clamp to physical limits
        if self.sensor_type == SensorType.HUMIDITY:
            value = max(0.0, min(100.0, value))
        elif self.sensor_type == SensorType.WATER_LEAK:
            value = 1.0 if value > 0.5 else 0.0
        elif self.sensor_type in (SensorType.GAS_CONCENTRATION, SensorType.SMOKE):
            value = max(0.0, value)

        return value

    def _evaluate_thresholds(self, value: float) -> tuple[SensorStatus, AlertLevel, str]:
        """Evaluate thresholds and return status, alert level, and message."""
        sensor_name = self.sensor_type.value.replace('_', ' ').title()

        # For sensors where danger is BELOW threshold (like pressure)
        if self.sensor_type == SensorType.PRESSURE:
            if value <= self.danger_threshold:
                return (SensorStatus.DANGER, AlertLevel.EMERGENCY,
                        f"SIMULATED ALERT: {sensor_name} critically low: "
                        f"{value:.1f} {self.unit} (threshold: {self.danger_threshold})")
            elif value <= self.alert_threshold:
                return (SensorStatus.ALERT, AlertLevel.WARNING,
                        f"SIMULATED ALERT: {sensor_name} low: "
                        f"{value:.1f} {self.unit} (threshold: {self.alert_threshold})")
            return (SensorStatus.NORMAL, AlertLevel.NONE, "")

        # For sensors where danger is ABOVE threshold
        if value >= self.danger_threshold:
            return (SensorStatus.DANGER, AlertLevel.EMERGENCY,
                    f"SIMULATED ALERT: {sensor_name} DANGER: "
                    f"{value:.1f} {self.unit} (danger threshold: {self.danger_threshold})")
        elif value >= self.alert_threshold:
            return (SensorStatus.ALERT, AlertLevel.CRITICAL,
                    f"SIMULATED ALERT: {sensor_name} exceeded threshold: "
                    f"{value:.1f} {self.unit} (alert threshold: {self.alert_threshold})")
        return (SensorStatus.NORMAL, AlertLevel.NONE, "")

    def inject_event(self, event_type: str, start_time: float,
                     duration: float, peak_value: float) -> None:
        """
        Inject a simulated event (gas leak, temperature spike, etc.).

        Args:
            event_type: Description of the event (e.g., "gas_leak").
            start_time: When the event begins (sim time).
            duration: Duration of the event in seconds.
            peak_value: Peak value during the event.
        """
        self.event_active = True
        self.event_type = event_type
        self.event_start_time = start_time
        self.event_end_time = start_time + duration
        self.event_target_value = peak_value

    def set_failed(self, failed: bool = True) -> None:
        """Set sensor to failed/recovered state."""
        self.is_failed = failed
        self.status = SensorStatus.FAILED if failed else SensorStatus.NORMAL

    def get_latest_reading(self) -> Optional[SensorReading]:
        """Get the most recent reading."""
        return self.reading_history[-1] if self.reading_history else None

    def get_state(self) -> dict:
        """Get current sensor state for dashboard."""
        latest = self.get_latest_reading()
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'device_id': self.device_id,
            'current_value': self.current_value * self.calibration_factor,
            'unit': self.unit,
            'status': self.status.name,
            'alert_threshold': self.alert_threshold,
            'danger_threshold': self.danger_threshold,
            'is_failed': self.is_failed,
            'event_active': self.event_active,
            'total_readings': self.sequence_counter,
            'last_reading_time': self.last_reading_time,
            'alert_count': len(self.alert_history),
        }


class SensorFactory:
    """Factory for creating sensors from configuration."""

    @staticmethod
    def create_sensor(sensor_type_str: str, config: dict,
                      device_id: str = "",
                      rng: random.Random = None) -> VirtualSensor:
        """Create a sensor from a type string and config dict."""
        type_map = {
            'gas': SensorType.GAS_CONCENTRATION,
            'gas_concentration': SensorType.GAS_CONCENTRATION,
            'temperature': SensorType.TEMPERATURE,
            'humidity': SensorType.HUMIDITY,
            'smoke': SensorType.SMOKE,
            'pressure': SensorType.PRESSURE,
            'water_leak': SensorType.WATER_LEAK,
            'appliance_status': SensorType.APPLIANCE_STATUS,
        }
        sensor_type = type_map.get(sensor_type_str, SensorType.TEMPERATURE)
        return VirtualSensor(sensor_type, config, device_id, rng)

    @staticmethod
    def create_kitchen_sensor_suite(device_id: str, sensor_configs: dict,
                                    rng: random.Random = None) -> list[VirtualSensor]:
        """Create a full suite of kitchen sensors for a device."""
        sensors = []
        for sensor_key, config in sensor_configs.items():
            sensor = SensorFactory.create_sensor(sensor_key, config, device_id, rng)
            sensors.append(sensor)
        return sensors
