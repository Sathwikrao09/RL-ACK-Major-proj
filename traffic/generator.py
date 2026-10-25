"""
Traffic Generator for the RL-ACK Smart Kitchen Simulator.

Generates message traffic across three tiers:
  CC  (Critical Control)        — 1-5 msg/s per device, 64B payload, 10ms deadline
  HBS (High-Bandwidth Streaming) — 15-30 msg/s per device, 1024B payload, 50ms deadline
  BET (Best-Effort Telemetry)   — 0.1-2 msg/s per device, 128B payload, 500ms deadline

Traffic composition (CC:20%, HBS:30%, BET:50%) determines how many devices
of each category exist. Arrival rates are PER DEVICE.
Classification is by message metadata (topic, source device category).
"""

import random
from typing import Optional

from sim.packet import Packet, TrafficTier, PacketType, create_data_packet, TIER_DEADLINES
from sim.node import NetworkNode, DeviceCategory


class TrafficConfig:
    """Traffic generation configuration."""

    def __init__(self, config: dict):
        self.composition = config.get('composition', {'CC': 0.2, 'HBS': 0.3, 'BET': 0.5})
        self.tiers = config.get('tiers', {})

        # Validate composition sums to 1.0
        total = sum(self.composition.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Traffic composition must sum to 1.0, got {total}. "
                f"Composition: {self.composition}"
            )

    def get_arrival_rate(self, tier: TrafficTier) -> float:
        """Get default arrival rate (msg/s) for a tier."""
        tier_config = self.tiers.get(tier.value, {})
        return tier_config.get('default_arrival_rate', 1.0)

    def get_payload_size(self, tier: TrafficTier) -> int:
        """Get payload size in bytes for a tier."""
        tier_config = self.tiers.get(tier.value, {})
        return tier_config.get('payload_size_bytes', 128)

    def get_device_counts(self, total_devices: int) -> dict[TrafficTier, int]:
        """
        Calculate number of devices per tier based on composition.

        Ensures at least 1 device per active tier and total matches.
        """
        counts = {}
        remaining = total_devices

        for tier_str, fraction in self.composition.items():
            tier = TrafficTier(tier_str)
            count = max(1, round(total_devices * fraction))
            counts[tier] = count

        # Adjust to match total
        total_assigned = sum(counts.values())
        if total_assigned > total_devices:
            # Remove from largest group
            largest = max(counts, key=counts.get)
            counts[largest] -= (total_assigned - total_devices)
        elif total_assigned < total_devices:
            # Add to largest group
            largest = max(counts, key=counts.get)
            counts[largest] += (total_devices - total_assigned)

        return counts


class TrafficClassifier:
    """
    Classifies messages into traffic tiers based on observable metadata.

    Classification uses topic, source device category, payload size,
    and message type — NOT a hardcoded mapping.
    """

    # Topic prefix patterns for classification
    TOPIC_PATTERNS = {
        TrafficTier.CC: ['control/', 'safety/', 'actuator/', 'emergency/', 'gas/alert'],
        TrafficTier.HBS: ['camera/', 'video/', 'stream/', 'sensor/high_rate/'],
        TrafficTier.BET: ['telemetry/', 'sensor/', 'environment/', 'status/'],
    }

    # Device category to primary tier mapping
    CATEGORY_MAP = {
        DeviceCategory.CRITICAL_CONTROL: TrafficTier.CC,
        DeviceCategory.STREAMING: TrafficTier.HBS,
        DeviceCategory.TELEMETRY: TrafficTier.BET,
    }

    @classmethod
    def classify(cls, topic: str = "", source_category: DeviceCategory = None,
                 payload_size: int = 0, message_type: str = "") -> TrafficTier:
        """
        Classify a message into a traffic tier.

        Priority: explicit topic pattern > device category > payload heuristic.

        Note: Classification is per-message, not permanently per device.
        A telemetry device can generate a CC message if it detects an alert.
        """
        # 1. Check topic patterns
        topic_lower = topic.lower()
        for tier, patterns in cls.TOPIC_PATTERNS.items():
            if any(topic_lower.startswith(p) for p in patterns):
                return tier

        # 2. Check message type (alerts from any device are CC)
        if message_type in ('alert', 'emergency', 'critical'):
            return TrafficTier.CC

        # 3. Check device category
        if source_category:
            mapped = cls.CATEGORY_MAP.get(source_category)
            if mapped:
                return mapped

        # 4. Payload size heuristic
        if payload_size <= 100:
            return TrafficTier.CC
        elif payload_size >= 512:
            return TrafficTier.HBS

        return TrafficTier.BET


class TrafficGenerator:
    """
    Generates message traffic for the simulation.

    Creates messages at configured arrival rates for each device, with
    proper traffic tier classification and sensor data attachment.
    """

    def __init__(self, config: dict, rng: random.Random = None):
        self.traffic_config = TrafficConfig(config)
        self.rng = rng or random.Random()

        # Statistics
        self.total_generated = 0
        self.generated_by_tier = {tier: 0 for tier in TrafficTier}
        self.generation_log: list[dict] = []

    def classify_message(self, topic: str = "", payload_size: int = 0,
                         source_category: DeviceCategory = None,
                         message_type: str = "") -> TrafficTier:
        """Classify a message into its traffic tier based on metadata."""
        return TrafficClassifier.classify(topic, source_category, payload_size, message_type)

    def compute_inter_arrival_time(self, tier: TrafficTier,
                                   utilization_target: float = 0.7) -> float:
        """
        Compute inter-arrival time in seconds for a traffic tier.

        Adjusts arrival rate to help achieve target utilization.
        The rate is PER DEVICE, not aggregate.
        """
        base_rate = self.traffic_config.get_arrival_rate(tier)

        # Scale rate by utilization target (higher util = higher rate)
        scaled_rate = base_rate * (utilization_target / 0.7)

        if scaled_rate <= 0:
            return 10.0

        # Exponential inter-arrival time
        return self.rng.expovariate(scaled_rate)

    def generate_message(self, source_node: NetworkNode,
                         dest_id: str, sim_time: float,
                         sensor_data: dict = None,
                         force_tier: TrafficTier = None) -> Packet:
        """
        Generate a single message from a source node.

        Args:
            source_node: The sending device node.
            dest_id: Destination node ID (typically edge router).
            sim_time: Current simulation time.
            sensor_data: Optional sensor reading to include.
            force_tier: Override automatic classification.

        Returns:
            A new data Packet ready for queuing.
        """
        # Determine traffic tier
        if force_tier:
            tier = force_tier
        else:
            # Classify based on device category and sensor data
            message_type = ""
            topic = f"telemetry/{source_node.node_id}"

            if sensor_data:
                alert_level = sensor_data.get('alert_level', 'NONE')
                if alert_level in ('CRITICAL', 'EMERGENCY'):
                    message_type = 'alert'
                    topic = f"safety/{source_node.node_id}/alert"
                elif sensor_data.get('sensor_type') in ('gas_concentration', 'smoke'):
                    topic = f"safety/{source_node.node_id}/{sensor_data['sensor_type']}"

            tier = TrafficClassifier.classify(
                topic=topic,
                source_category=source_node.category,
                payload_size=self.traffic_config.get_payload_size(
                    TrafficTier(source_node.category.name.split('_')[0])
                    if hasattr(source_node.category, 'name') else TrafficTier.BET
                ),
                message_type=message_type,
            )

        # Create packet
        payload_size = self.traffic_config.get_payload_size(tier)
        topic = self._generate_topic(source_node, tier, sensor_data)

        packet = create_data_packet(
            source_id=source_node.node_id,
            dest_id=dest_id,
            tier=tier,
            payload_size=payload_size,
            gen_time=sim_time,
            topic=topic,
            sensor_data=sensor_data or {},
        )

        # Update stats
        self.total_generated += 1
        self.generated_by_tier[tier] += 1
        source_node.messages_generated += 1
        source_node.tier_counts[tier] += 1

        return packet

    def _generate_topic(self, node: NetworkNode, tier: TrafficTier,
                        sensor_data: dict = None) -> str:
        """Generate an MQTT-like topic string."""
        if sensor_data and sensor_data.get('alert_level') in ('CRITICAL', 'EMERGENCY'):
            return f"safety/{node.node_id}/{sensor_data.get('sensor_type', 'alert')}"

        topic_prefixes = {
            TrafficTier.CC: 'control',
            TrafficTier.HBS: 'stream',
            TrafficTier.BET: 'telemetry',
        }
        prefix = topic_prefixes.get(tier, 'data')

        if sensor_data:
            return f"{prefix}/{node.node_id}/{sensor_data.get('sensor_type', 'sensor')}"
        return f"{prefix}/{node.node_id}"

    def schedule_next_message(self, tier: TrafficTier,
                              utilization_target: float = 0.7) -> float:
        """Get the delay until the next message should be generated."""
        return self.compute_inter_arrival_time(tier, utilization_target)

    def get_stats(self) -> dict:
        """Get traffic generation statistics."""
        return {
            'total_generated': self.total_generated,
            'by_tier': {tier.value: count for tier, count in self.generated_by_tier.items()},
            'composition_actual': {
                tier.value: count / max(1, self.total_generated)
                for tier, count in self.generated_by_tier.items()
            },
        }
