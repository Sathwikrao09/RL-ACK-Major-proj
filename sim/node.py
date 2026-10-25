"""
Network Node models for the RL-ACK Smart Kitchen Simulator.

Defines IoT device nodes (sensors/actuators) and the edge-router node
that hosts the RL-ACK controller. Each node has a transmit queue,
energy model, and device-level state tracking.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .packet import Packet, DeliveryStatus, TrafficTier


class NodeRole(Enum):
    """Role of a network node."""
    IOT_DEVICE = auto()    # Sensor/actuator device
    EDGE_ROUTER = auto()   # AP / Edge router / RL-ACK controller host
    SUBSCRIBER = auto()    # MQTT subscriber node


class DeviceCategory(Enum):
    """Device category determines which traffic tiers a device generates."""
    CRITICAL_CONTROL = auto()     # Gas sensors, safety actuators
    STREAMING = auto()            # Cameras, high-rate sensors
    TELEMETRY = auto()            # Temperature, humidity, routine monitoring
    MIXED = auto()                # Generates multiple traffic types


@dataclass
class EnergyModel:
    """
    Simplified energy model for IoT devices.

    Energy values are modeled estimates, not measured hardware values.
    Labeled as modeled energy throughout.
    """
    initial_energy_mj: float = 10000.0   # 10 J initial battery
    current_energy_mj: float = 10000.0
    tx_power_mj_per_byte: float = 0.001  # Transmit energy per byte
    rx_power_mj_per_byte: float = 0.0005 # Receive energy per byte
    idle_power_mj_per_s: float = 0.01    # Idle consumption
    total_consumed_mj: float = 0.0

    def consume_tx(self, size_bytes: int) -> float:
        """Consume energy for transmitting. Returns mJ consumed."""
        cost = size_bytes * self.tx_power_mj_per_byte
        self.current_energy_mj -= cost
        self.total_consumed_mj += cost
        return cost

    def consume_rx(self, size_bytes: int) -> float:
        """Consume energy for receiving. Returns mJ consumed."""
        cost = size_bytes * self.rx_power_mj_per_byte
        self.current_energy_mj -= cost
        self.total_consumed_mj += cost
        return cost

    def consume_idle(self, duration_s: float) -> float:
        """Consume idle energy over a duration. Returns mJ consumed."""
        cost = duration_s * self.idle_power_mj_per_s
        self.current_energy_mj -= cost
        self.total_consumed_mj += cost
        return cost

    @property
    def remaining_fraction(self) -> float:
        """Fraction of energy remaining (0 to 1)."""
        if self.initial_energy_mj <= 0:
            return 0.0
        return max(0.0, self.current_energy_mj / self.initial_energy_mj)


class TransmitQueue:
    """
    FIFO transmit queue with size limit and occupancy tracking.

    Tracks enqueue/dequeue events for metric computation.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._queue: deque[Packet] = deque()
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.total_dropped = 0
        self.occupancy_samples: list[tuple[float, int]] = []

    def enqueue(self, packet: Packet, sim_time: float) -> bool:
        """
        Enqueue a packet. Returns False if queue is full (packet dropped).
        """
        if len(self._queue) >= self.max_size:
            self.total_dropped += 1
            packet.status = DeliveryStatus.DROPPED
            return False
        packet.enqueue_time = sim_time
        packet.status = DeliveryStatus.PENDING
        self._queue.append(packet)
        self.total_enqueued += 1
        self.occupancy_samples.append((sim_time, len(self._queue)))
        return True

    def dequeue(self, sim_time: float) -> Optional[Packet]:
        """Dequeue the next packet, or None if empty."""
        if self._queue:
            pkt = self._queue.popleft()
            self.total_dequeued += 1
            self.occupancy_samples.append((sim_time, len(self._queue)))
            return pkt
        return None

    def peek(self) -> Optional[Packet]:
        """Look at the front packet without removing it."""
        return self._queue[0] if self._queue else None

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def occupancy_fraction(self) -> float:
        """Current queue occupancy as a fraction (0 to 1)."""
        if self.max_size <= 0:
            return 0.0
        return len(self._queue) / self.max_size

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self.max_size

    def clear(self) -> None:
        self._queue.clear()


class NetworkNode:
    """
    A network node in the smart kitchen simulation.

    Can be an IoT device (sensor/actuator), an edge router, or a subscriber.
    """

    def __init__(self, node_id: str, role: NodeRole,
                 category: DeviceCategory = DeviceCategory.TELEMETRY,
                 position: tuple[float, float] = (0.0, 0.0),
                 max_queue_size: int = 200,
                 rng: random.Random = None):
        self.node_id = node_id
        self.role = role
        self.category = category
        self.position = position
        self.rng = rng or random.Random()

        # Subsystems
        self.tx_queue = TransmitQueue(max_size=max_queue_size)
        self.energy = EnergyModel()

        # State
        self.is_active = True
        self.is_transmitting = False

        # Traffic tracking
        self.messages_generated = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_failed = 0
        self.acks_received = 0

        # Per-tier tracking
        self.tier_counts = {tier: 0 for tier in TrafficTier}

        # Assigned sensors (for IoT devices)
        self.sensor_ids: list[str] = []

        # Latency tracking for rolling average
        self._recent_latencies: deque[float] = deque(maxlen=50)

        # Pending messages awaiting ACK
        self.pending_acks: dict[str, Packet] = {}

    def distance_to(self, other: 'NetworkNode') -> float:
        """Euclidean distance to another node."""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def record_latency(self, latency: float) -> None:
        """Record a delivery latency for rolling average."""
        self._recent_latencies.append(latency)

    @property
    def rolling_avg_latency(self) -> float:
        """Rolling average of recent latencies."""
        if not self._recent_latencies:
            return 0.0
        return sum(self._recent_latencies) / len(self._recent_latencies)

    def get_state_vector(self) -> dict:
        """Get node state for monitoring/dashboard."""
        return {
            'node_id': self.node_id,
            'role': self.role.name,
            'category': self.category.name,
            'position': self.position,
            'is_active': self.is_active,
            'queue_occupancy': self.tx_queue.occupancy_fraction,
            'queue_size': self.tx_queue.size,
            'energy_remaining': self.energy.remaining_fraction,
            'messages_generated': self.messages_generated,
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'rolling_latency': self.rolling_avg_latency,
        }
