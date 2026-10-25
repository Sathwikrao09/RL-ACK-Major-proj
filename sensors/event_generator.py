"""
Sensor Event Generator for the RL-ACK Smart Kitchen Simulator.

Schedules and manages simulated events such as gas leaks, temperature
spikes, smoke events, and sensor failures. All events are synthetic
simulations, not real-world safety events.
"""

import random
from dataclasses import dataclass
from typing import Optional

from .sensor_models import VirtualSensor, SensorType


@dataclass
class ScheduledEvent:
    """A scheduled sensor event."""
    event_id: str
    sensor_id: str
    event_type: str
    start_time: float
    duration: float
    peak_value: float
    triggered: bool = False
    completed: bool = False


class EventGenerator:
    """
    Generates and schedules sensor events based on configurable probabilities.

    Events are pre-scheduled at initialization or can be injected dynamically.
    """

    def __init__(self, event_configs: dict, rng: random.Random = None):
        """
        Args:
            event_configs: Event configuration from default_config.json['sensor_events'].
            rng: Random number generator for reproducibility.
        """
        self.event_configs = event_configs
        self.rng = rng or random.Random()
        self.scheduled_events: list[ScheduledEvent] = []
        self.active_events: list[ScheduledEvent] = []
        self.event_log: list[dict] = []
        self._event_counter = 0

    def generate_events_for_sensor(self, sensor: VirtualSensor,
                                   sim_duration: float) -> list[ScheduledEvent]:
        """
        Pre-generate events for a sensor over the entire simulation duration.

        Uses Poisson process: events occur at a rate defined by
        probability_per_hour, distributed uniformly across the simulation.
        """
        events = []

        # Map sensor type to event config
        event_type_map = {
            SensorType.GAS_CONCENTRATION: 'gas_leak',
            SensorType.TEMPERATURE: 'temperature_spike',
            SensorType.SMOKE: 'smoke_event',
        }

        event_key = event_type_map.get(sensor.sensor_type)
        if event_key and event_key in self.event_configs:
            config = self.event_configs[event_key]
            prob_per_hour = config.get('probability_per_hour', 0.1)
            duration_range = config.get('duration_range_s', [30, 120])

            # Determine peak value based on event type
            if event_key == 'gas_leak':
                peak = config.get('peak_ppm', 800)
            elif event_key == 'temperature_spike':
                peak = config.get('peak_celsius', 90)
            elif event_key == 'smoke_event':
                peak = config.get('peak_obscuration', 15.0)
            else:
                peak = sensor.danger_threshold * 1.5

            # Generate events using Poisson process
            sim_hours = sim_duration / 3600.0
            expected_events = prob_per_hour * sim_hours
            num_events = self.rng.poisson(expected_events) if hasattr(self.rng, 'poisson') else \
                         int(expected_events) + (1 if self.rng.random() < (expected_events % 1) else 0)

            for _ in range(num_events):
                self._event_counter += 1
                duration = self.rng.uniform(duration_range[0], duration_range[1])
                start_time = self.rng.uniform(10.0, max(11.0, sim_duration - duration - 10))

                evt = ScheduledEvent(
                    event_id=f"evt_{self._event_counter:04d}",
                    sensor_id=sensor.sensor_id,
                    event_type=event_key,
                    start_time=start_time,
                    duration=duration,
                    peak_value=peak,
                )
                events.append(evt)
                self.scheduled_events.append(evt)

        # Sensor failure events (applicable to all sensor types)
        failure_config = self.event_configs.get('sensor_failure', {})
        if failure_config:
            prob_per_hour = failure_config.get('probability_per_hour', 0.02)
            duration_range = failure_config.get('duration_range_s', [60, 600])
            sim_hours = sim_duration / 3600.0
            expected = prob_per_hour * sim_hours
            num_failures = int(expected) + (1 if self.rng.random() < (expected % 1) else 0)

            for _ in range(num_failures):
                self._event_counter += 1
                duration = self.rng.uniform(duration_range[0], duration_range[1])
                start_time = self.rng.uniform(30.0, max(31.0, sim_duration - duration - 10))

                evt = ScheduledEvent(
                    event_id=f"fail_{self._event_counter:04d}",
                    sensor_id=sensor.sensor_id,
                    event_type='sensor_failure',
                    start_time=start_time,
                    duration=duration,
                    peak_value=0.0,
                )
                events.append(evt)
                self.scheduled_events.append(evt)

        return events

    def check_and_trigger_events(self, sensors: dict[str, VirtualSensor],
                                 sim_time: float) -> list[dict]:
        """
        Check for events that should be triggered or completed at sim_time.

        Args:
            sensors: Dict of sensor_id -> VirtualSensor.
            sim_time: Current simulation time.

        Returns:
            List of event notifications.
        """
        notifications = []

        for evt in self.scheduled_events:
            if evt.completed:
                continue

            sensor = sensors.get(evt.sensor_id)
            if sensor is None:
                continue

            # Trigger event
            if not evt.triggered and sim_time >= evt.start_time:
                evt.triggered = True
                self.active_events.append(evt)

                if evt.event_type == 'sensor_failure':
                    sensor.set_failed(True)
                    notification = {
                        'time': sim_time,
                        'event_id': evt.event_id,
                        'event_type': evt.event_type,
                        'sensor_id': evt.sensor_id,
                        'action': 'start',
                        'message': f"SIMULATED: Sensor {evt.sensor_id} has failed",
                    }
                else:
                    sensor.inject_event(
                        evt.event_type, evt.start_time,
                        evt.duration, evt.peak_value
                    )
                    notification = {
                        'time': sim_time,
                        'event_id': evt.event_id,
                        'event_type': evt.event_type,
                        'sensor_id': evt.sensor_id,
                        'action': 'start',
                        'message': f"SIMULATED: {evt.event_type} started on {evt.sensor_id} "
                                   f"(peak: {evt.peak_value})",
                    }

                notifications.append(notification)
                self.event_log.append(notification)

            # Complete event
            if evt.triggered and not evt.completed and \
               sim_time >= evt.start_time + evt.duration:
                evt.completed = True
                if evt in self.active_events:
                    self.active_events.remove(evt)

                if evt.event_type == 'sensor_failure':
                    sensor.set_failed(False)

                notification = {
                    'time': sim_time,
                    'event_id': evt.event_id,
                    'event_type': evt.event_type,
                    'sensor_id': evt.sensor_id,
                    'action': 'end',
                    'message': f"SIMULATED: {evt.event_type} ended on {evt.sensor_id}",
                }
                notifications.append(notification)
                self.event_log.append(notification)

        return notifications

    def inject_gas_leak(self, sensor: VirtualSensor, start_time: float,
                        duration: float = 60.0, peak_ppm: float = 800.0) -> ScheduledEvent:
        """Manually inject a gas leak event for demonstration."""
        self._event_counter += 1
        evt = ScheduledEvent(
            event_id=f"demo_gas_{self._event_counter:04d}",
            sensor_id=sensor.sensor_id,
            event_type='gas_leak',
            start_time=start_time,
            duration=duration,
            peak_value=peak_ppm,
        )
        self.scheduled_events.append(evt)
        return evt

    def inject_temperature_spike(self, sensor: VirtualSensor, start_time: float,
                                 duration: float = 120.0,
                                 peak_celsius: float = 90.0) -> ScheduledEvent:
        """Manually inject a temperature spike event."""
        self._event_counter += 1
        evt = ScheduledEvent(
            event_id=f"demo_temp_{self._event_counter:04d}",
            sensor_id=sensor.sensor_id,
            event_type='temperature_spike',
            start_time=start_time,
            duration=duration,
            peak_value=peak_celsius,
        )
        self.scheduled_events.append(evt)
        return evt

    def get_event_summary(self) -> dict:
        """Get a summary of all events for reporting."""
        return {
            'total_scheduled': len(self.scheduled_events),
            'total_triggered': sum(1 for e in self.scheduled_events if e.triggered),
            'total_completed': sum(1 for e in self.scheduled_events if e.completed),
            'currently_active': len(self.active_events),
            'event_types': {},
            'log_entries': len(self.event_log),
        }
