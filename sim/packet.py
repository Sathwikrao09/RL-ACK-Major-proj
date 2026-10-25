"""
Packet and Message model for the RL-ACK simulator.

Every message carries full metadata: ID, source, traffic tier, timestamps,
payload size, deadline, and delivery tracking. ACK packets, control packets,
and retransmissions are distinct packet types with different sizes.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid


class TrafficTier(Enum):
    """Traffic classification tiers from the paper."""
    CC = "CC"    # Critical Control
    HBS = "HBS"  # High-Bandwidth Streaming
    BET = "BET"  # Best-Effort Telemetry


class PacketType(Enum):
    """Types of packets in the simulation."""
    DATA = auto()          # Original data message
    ACK = auto()           # Acknowledgment packet
    LOCAL_ACK = auto()     # Local/proxy acknowledgment
    PUBACK = auto()        # QoS 1 PUBACK
    PUBREC = auto()        # QoS 2 step 1
    PUBREL = auto()        # QoS 2 step 2
    PUBCOMP = auto()       # QoS 2 step 3
    RETRANSMIT = auto()    # Retransmitted data
    NACK = auto()          # Negative acknowledgment (selective retransmit)
    CONTROL = auto()       # Generic control packet


class DeliveryStatus(Enum):
    """Delivery outcome of a message."""
    PENDING = auto()
    IN_TRANSIT = auto()
    DELIVERED = auto()
    ACKED = auto()
    LOST = auto()
    TIMED_OUT = auto()
    DEADLINE_MISSED = auto()
    DROPPED = auto()


# Header sizes for different packet types (bytes)
HEADER_SIZES = {
    PacketType.DATA: 40,        # IP(20) + TCP(20) + MQTT-like(variable)
    PacketType.ACK: 20,         # Minimal ACK
    PacketType.LOCAL_ACK: 16,   # Local proxy ACK (shorter path)
    PacketType.PUBACK: 24,      # MQTT PUBACK equivalent
    PacketType.PUBREC: 24,      # QoS 2 PUBREC
    PacketType.PUBREL: 24,      # QoS 2 PUBREL
    PacketType.PUBCOMP: 24,     # QoS 2 PUBCOMP
    PacketType.RETRANSMIT: 44,  # DATA header + retransmit flag
    PacketType.NACK: 28,        # NACK with sequence info
    PacketType.CONTROL: 20,     # Generic control
}

# Deadline per traffic tier (seconds)
TIER_DEADLINES = {
    TrafficTier.CC: 0.010,    # 10 ms
    TrafficTier.HBS: 0.050,   # 50 ms
    TrafficTier.BET: 0.500,   # 500 ms
}


@dataclass
class Packet:
    """
    A network packet in the simulation.

    This can be a data message, an ACK, a control packet, or a retransmission.
    Every packet tracks its lifecycle timestamps for metric computation.
    """
    # Identity
    packet_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    message_id: str = ""       # Parent message ID (for ACKs and retransmits)
    packet_type: PacketType = PacketType.DATA
    sequence_num: int = 0

    # Source/destination
    source_id: str = ""
    destination_id: str = ""

    # Traffic classification
    traffic_tier: TrafficTier = TrafficTier.BET
    topic: str = ""            # MQTT-like topic string

    # Payload
    payload_size_bytes: int = 128
    header_size_bytes: int = 40

    # Timing
    generation_time: float = 0.0     # When the message was created
    enqueue_time: float = 0.0        # When enqueued at source
    transmit_start_time: float = 0.0 # When transmission began
    transmit_end_time: float = 0.0   # When transmission completed
    arrival_time: float = 0.0        # When received at destination
    ack_send_time: float = 0.0       # When ACK was sent
    ack_arrive_time: float = 0.0     # When ACK was received at source
    deadline: float = 0.0            # Absolute deadline (generation_time + tier_deadline)

    # Delivery tracking
    status: DeliveryStatus = DeliveryStatus.PENDING
    retransmit_count: int = 0
    max_retransmits: int = 0
    ack_action: int = -1  # Which ACK action was applied (0-4)

    # Energy tracking
    energy_consumed_mj: float = 0.0

    # Sensor data (for DATA packets carrying sensor readings)
    sensor_data: dict = field(default_factory=dict)

    @property
    def total_size_bytes(self) -> int:
        """Total packet size including headers."""
        return self.payload_size_bytes + self.header_size_bytes

    @property
    def end_to_end_latency(self) -> Optional[float]:
        """End-to-end latency in seconds, or None if not yet delivered."""
        if self.status in (DeliveryStatus.DELIVERED, DeliveryStatus.ACKED):
            return self.arrival_time - self.generation_time
        return None

    @property
    def queue_delay(self) -> Optional[float]:
        """Time spent in queue (seconds)."""
        if self.transmit_start_time > 0 and self.enqueue_time > 0:
            return self.transmit_start_time - self.enqueue_time
        return None

    @property
    def transmission_delay(self) -> Optional[float]:
        """Time for transmission (seconds)."""
        if self.transmit_end_time > 0 and self.transmit_start_time > 0:
            return self.transmit_end_time - self.transmit_start_time
        return None

    @property
    def is_deadline_compliant(self) -> Optional[bool]:
        """Whether the message met its deadline."""
        latency = self.end_to_end_latency
        if latency is None:
            return None
        deadline_duration = TIER_DEADLINES.get(self.traffic_tier, 1.0)
        return latency <= deadline_duration

    @property
    def is_expired(self) -> bool:
        """Whether the message deadline has passed."""
        return self.status == DeliveryStatus.DEADLINE_MISSED

    def set_deadline(self) -> None:
        """Set the absolute deadline based on generation time and tier."""
        self.deadline = self.generation_time + TIER_DEADLINES.get(
            self.traffic_tier, 1.0
        )

    def mark_delivered(self, arrival_time: float) -> None:
        """Mark the packet as delivered."""
        self.arrival_time = arrival_time
        self.status = DeliveryStatus.DELIVERED

    def mark_acked(self, ack_time: float) -> None:
        """Mark the packet as acknowledged."""
        self.ack_arrive_time = ack_time
        self.status = DeliveryStatus.ACKED

    def mark_lost(self) -> None:
        """Mark the packet as lost."""
        self.status = DeliveryStatus.LOST

    def mark_timed_out(self) -> None:
        """Mark the packet as timed out."""
        self.status = DeliveryStatus.TIMED_OUT

    def mark_deadline_missed(self) -> None:
        """Mark the packet as having missed its deadline."""
        self.status = DeliveryStatus.DEADLINE_MISSED

    def to_dict(self) -> dict:
        """Serialize to dict for logging/export."""
        return {
            'packet_id': self.packet_id,
            'message_id': self.message_id,
            'packet_type': self.packet_type.name,
            'source_id': self.source_id,
            'destination_id': self.destination_id,
            'traffic_tier': self.traffic_tier.value,
            'topic': self.topic,
            'payload_size': self.payload_size_bytes,
            'total_size': self.total_size_bytes,
            'generation_time': self.generation_time,
            'arrival_time': self.arrival_time,
            'deadline': self.deadline,
            'status': self.status.name,
            'latency': self.end_to_end_latency,
            'queue_delay': self.queue_delay,
            'transmission_delay': self.transmission_delay,
            'deadline_compliant': self.is_deadline_compliant,
            'retransmit_count': self.retransmit_count,
            'ack_action': self.ack_action,
            'energy_mj': self.energy_consumed_mj,
            'sensor_data': self.sensor_data,
        }


def create_data_packet(source_id: str, dest_id: str, tier: TrafficTier,
                       payload_size: int, gen_time: float,
                       topic: str = "", sensor_data: dict = None) -> Packet:
    """Factory function to create a properly initialized data packet."""
    pkt = Packet(
        message_id=uuid.uuid4().hex[:12],
        packet_type=PacketType.DATA,
        source_id=source_id,
        destination_id=dest_id,
        traffic_tier=tier,
        topic=topic,
        payload_size_bytes=payload_size,
        header_size_bytes=HEADER_SIZES[PacketType.DATA],
        generation_time=gen_time,
        sensor_data=sensor_data or {},
    )
    pkt.set_deadline()
    return pkt


def create_ack_packet(original: Packet, ack_type: PacketType,
                      send_time: float) -> Packet:
    """Create an ACK/control packet in response to a data packet."""
    return Packet(
        message_id=original.message_id,
        packet_type=ack_type,
        source_id=original.destination_id,
        destination_id=original.source_id,
        traffic_tier=original.traffic_tier,
        topic=original.topic,
        payload_size_bytes=0,
        header_size_bytes=HEADER_SIZES.get(ack_type, 20),
        generation_time=send_time,
        ack_action=original.ack_action,
    )


def create_retransmit_packet(original: Packet, retransmit_time: float) -> Packet:
    """Create a retransmission of an original data packet."""
    pkt = Packet(
        message_id=original.message_id,
        packet_type=PacketType.RETRANSMIT,
        source_id=original.source_id,
        destination_id=original.destination_id,
        traffic_tier=original.traffic_tier,
        topic=original.topic,
        payload_size_bytes=original.payload_size_bytes,
        header_size_bytes=HEADER_SIZES[PacketType.RETRANSMIT],
        generation_time=original.generation_time,
        enqueue_time=retransmit_time,
        deadline=original.deadline,
        retransmit_count=original.retransmit_count + 1,
        max_retransmits=original.max_retransmits,
        ack_action=original.ack_action,
        sensor_data=original.sensor_data,
    )
    pkt.status = DeliveryStatus.PENDING
    return pkt
