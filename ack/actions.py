"""
Five ACK Action Implementations for the RL-ACK Framework.

Each action produces DISTINCT network behavior — different control packets,
handshake sequences, retransmission logic, delays, and energy costs.

Actions:
  a0: No ACK (QoS 0-like) — fire-and-forget
  a1: Local/proxy ACK — edge router immediately ACKs, forwards independently
  a2: PUBACK (QoS 1-like) — end-to-end ACK with retransmission
  a3: Local ACK + Selective Retransmission — local ACK, edge monitors & selectively retransmits
  a4: QoS 2-like Multi-step Handshake — 4-step exactly-once delivery

These are application-level ACK abstractions, not native MQTT protocol
implementations. The distinction is documented throughout.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from sim.packet import (
    Packet, PacketType, DeliveryStatus, TrafficTier,
    create_ack_packet, create_retransmit_packet, HEADER_SIZES
)


class AckAction(IntEnum):
    """The five ACK actions from the paper."""
    A0_NO_ACK = 0
    A1_LOCAL_ACK = 1
    A2_PUBACK = 2
    A3_LOCAL_SELECTIVE = 3
    A4_QOS2_HANDSHAKE = 4


@dataclass
class AckOutcome:
    """Result of applying an ACK action to a message."""
    action: AckAction
    original_packet: Packet
    control_packets_generated: list[Packet]
    total_control_bytes: int
    total_control_packets: int
    handshake_delay_ms: float
    energy_cost_mj: float
    retransmission_limit: int
    requires_end_to_end_ack: bool
    requires_local_ack: bool
    requires_forwarding: bool
    description: str


# Maximum overhead per action (bytes) — used for reward normalization
MAX_OVERHEAD_BYTES = {
    AckAction.A0_NO_ACK: 0,
    AckAction.A1_LOCAL_ACK: HEADER_SIZES[PacketType.LOCAL_ACK],
    AckAction.A2_PUBACK: HEADER_SIZES[PacketType.PUBACK] * 4,  # With retransmits
    AckAction.A3_LOCAL_SELECTIVE: (HEADER_SIZES[PacketType.LOCAL_ACK] +
                                   HEADER_SIZES[PacketType.NACK] * 4 +
                                   HEADER_SIZES[PacketType.RETRANSMIT] * 4),
    AckAction.A4_QOS2_HANDSHAKE: (HEADER_SIZES[PacketType.PUBREC] +
                                   HEADER_SIZES[PacketType.PUBREL] +
                                   HEADER_SIZES[PacketType.PUBCOMP]) * 6,
}

# Absolute max overhead across all actions (for normalization)
GLOBAL_MAX_OVERHEAD = max(MAX_OVERHEAD_BYTES.values())


class AckActionHandler:
    """
    Handles the execution of ACK actions on messages.

    Each action method generates the appropriate control packets and
    sets up retransmission state. The simulation engine schedules
    the actual packet events.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: ACK action config from default_config.json['ack_actions'].
        """
        self.config = config or {}
        self.action_stats = {a: {'applied': 0, 'control_packets': 0,
                                  'retransmits': 0, 'energy_mj': 0.0}
                             for a in AckAction}

    def get_max_retries(self, action: AckAction) -> int:
        """Get the maximum retry limit for an action."""
        limits = {
            AckAction.A0_NO_ACK: 0,
            AckAction.A1_LOCAL_ACK: 0,
            AckAction.A2_PUBACK: self.config.get('a2', {}).get('retransmission_limit', 3),
            AckAction.A3_LOCAL_SELECTIVE: self.config.get('a3', {}).get('retransmission_limit', 4),
            AckAction.A4_QOS2_HANDSHAKE: self.config.get('a4', {}).get('retransmission_limit', 5),
        }
        return limits.get(action, 0)

    def get_control_packet_count(self, action: AckAction) -> int:
        """Get the number of control packets generated per message."""
        counts = {
            AckAction.A0_NO_ACK: 0,
            AckAction.A1_LOCAL_ACK: 1,
            AckAction.A2_PUBACK: 1,
            AckAction.A3_LOCAL_SELECTIVE: 1,
            AckAction.A4_QOS2_HANDSHAKE: 3,
        }
        return counts.get(action, 0)

    def get_handshake_delay(self, action: AckAction) -> float:
        """Get handshake delay in milliseconds."""
        delays = {
            AckAction.A0_NO_ACK: 0.0,
            AckAction.A1_LOCAL_ACK: self.config.get('a1', {}).get('handshake_delay_ms', 1.0),
            AckAction.A2_PUBACK: self.config.get('a2', {}).get('handshake_delay_ms', 2.0),
            AckAction.A3_LOCAL_SELECTIVE: self.config.get('a3', {}).get('handshake_delay_ms', 3.0),
            AckAction.A4_QOS2_HANDSHAKE: self.config.get('a4', {}).get('handshake_delay_ms', 5.0),
        }
        return delays.get(action, 0.0)

    def get_energy_cost_mj(self, action: AckAction) -> float:
        """Get modeled energy cost in millijoules."""
        costs = {
            AckAction.A0_NO_ACK: self.config.get('a0', {}).get('energy_cost_mj', 0.05),
            AckAction.A1_LOCAL_ACK: self.config.get('a1', {}).get('energy_cost_mj', 0.2),
            AckAction.A2_PUBACK: self.config.get('a2', {}).get('energy_cost_mj', 0.3),
            AckAction.A3_LOCAL_SELECTIVE: self.config.get('a3', {}).get('energy_cost_mj', 0.35),
            AckAction.A4_QOS2_HANDSHAKE: self.config.get('a4', {}).get('energy_cost_mj', 0.8),
        }
        return costs.get(action, 0.0)

    def get_overhead_bytes(self, action: AckAction) -> int:
        """Get baseline overhead in bytes."""
        return MAX_OVERHEAD_BYTES.get(action, 0)

    def apply_action(self, action: AckAction, packet: Packet,
                     sim_time: float) -> AckOutcome:
        """
        Apply an ACK action to a packet.

        This is the main dispatch method. It calls the specific handler
        for each action and returns the outcome including any generated
        control packets.
        """
        packet.ack_action = action.value
        self.action_stats[action]['applied'] += 1

        handlers = {
            AckAction.A0_NO_ACK: self._handle_a0,
            AckAction.A1_LOCAL_ACK: self._handle_a1,
            AckAction.A2_PUBACK: self._handle_a2,
            AckAction.A3_LOCAL_SELECTIVE: self._handle_a3,
            AckAction.A4_QOS2_HANDSHAKE: self._handle_a4,
        }

        handler = handlers.get(action, self._handle_a0)
        return handler(packet, sim_time)

    def _handle_a0(self, packet: Packet, sim_time: float) -> AckOutcome:
        """
        a0: No ACK (QoS 0-like).

        Fire-and-forget. No control packets. No retransmission.
        Lowest overhead but no delivery guarantee.

        Packet flow: Publisher -> Broker (no ACK returned)
        """
        packet.max_retransmits = 0

        return AckOutcome(
            action=AckAction.A0_NO_ACK,
            original_packet=packet,
            control_packets_generated=[],
            total_control_bytes=0,
            total_control_packets=0,
            handshake_delay_ms=0.0,
            energy_cost_mj=0.0,
            retransmission_limit=0,
            requires_end_to_end_ack=False,
            requires_local_ack=False,
            requires_forwarding=False,
            description="No ACK: fire-and-forget, no delivery guarantee",
        )

    def _handle_a1(self, packet: Packet, sim_time: float) -> AckOutcome:
        """
        a1: Local/Proxy ACK.

        The edge router immediately sends a LOCAL ACK to the publisher
        upon receiving the data packet, before forwarding to subscribers.
        The publisher considers delivery complete upon local ACK.
        No retransmission from publisher.

        Packet flow:
          Publisher -> Edge Router -> [LOCAL_ACK -> Publisher]
          Edge Router -> Subscriber (forwarding, independent of ACK)
        """
        config = self.config.get('a1', {})
        packet.max_retransmits = 0

        # Generate local ACK packet
        local_ack = create_ack_packet(packet, PacketType.LOCAL_ACK, sim_time)

        self.action_stats[AckAction.A1_LOCAL_ACK]['control_packets'] += 1
        energy = config.get('energy_cost_mj', 0.1)
        self.action_stats[AckAction.A1_LOCAL_ACK]['energy_mj'] += energy

        return AckOutcome(
            action=AckAction.A1_LOCAL_ACK,
            original_packet=packet,
            control_packets_generated=[local_ack],
            total_control_bytes=local_ack.total_size_bytes,
            total_control_packets=1,
            handshake_delay_ms=config.get('handshake_delay_ms', 0.5),
            energy_cost_mj=energy,
            retransmission_limit=0,
            requires_end_to_end_ack=False,
            requires_local_ack=True,
            requires_forwarding=True,
            description="Local ACK: edge router ACKs immediately, forwards independently",
        )

    def _handle_a2(self, packet: Packet, sim_time: float) -> AckOutcome:
        """
        a2: PUBACK (QoS 1-like).

        End-to-end acknowledgment. The broker sends PUBACK after receiving
        and processing the PUBLISH message. Publisher retransmits if no
        PUBACK received within timeout (up to 3 retries).

        Packet flow:
          Publisher -> Broker
          Broker -> [PUBACK -> Publisher]
          If no PUBACK: Publisher retransmits (up to 3 times)
        """
        config = self.config.get('a2', {})
        retransmit_limit = config.get('retransmission_limit', 3)
        packet.max_retransmits = retransmit_limit

        # Generate PUBACK packet
        puback = create_ack_packet(packet, PacketType.PUBACK, sim_time)

        self.action_stats[AckAction.A2_PUBACK]['control_packets'] += 1
        energy = config.get('energy_cost_mj', 0.3)
        self.action_stats[AckAction.A2_PUBACK]['energy_mj'] += energy

        return AckOutcome(
            action=AckAction.A2_PUBACK,
            original_packet=packet,
            control_packets_generated=[puback],
            total_control_bytes=puback.total_size_bytes,
            total_control_packets=1,
            handshake_delay_ms=config.get('handshake_delay_ms', 2.0),
            energy_cost_mj=energy,
            retransmission_limit=retransmit_limit,
            requires_end_to_end_ack=True,
            requires_local_ack=False,
            requires_forwarding=False,
            description="PUBACK: end-to-end ACK with up to 3 retransmissions",
        )

    def _handle_a3(self, packet: Packet, sim_time: float) -> AckOutcome:
        """
        a3: Local ACK + Selective Retransmission.

        The edge router sends a LOCAL ACK immediately, then forwards the
        message to subscribers. If forwarding fails, the edge router
        selectively retransmits to the specific failed subscribers
        (up to 4 retries). The publisher gets the local ACK quickly,
        while the edge router handles reliability.

        This DIFFERS from a1 (local ACK only) because a3 actively monitors
        forwarding outcomes and retransmits selectively.

        Packet flow:
          Publisher -> Edge Router -> [LOCAL_ACK -> Publisher]
          Edge Router -> Subscriber
          If forwarding fails:
            Edge Router -> [NACK check] -> Selective retransmit to failed subscribers
            (up to 4 retries per subscriber)
        """
        config = self.config.get('a3', {})
        retransmit_limit = config.get('retransmission_limit', 4)
        packet.max_retransmits = retransmit_limit

        # Generate local ACK + prepare for selective retransmission
        local_ack = create_ack_packet(packet, PacketType.LOCAL_ACK, sim_time)

        # The NACK will be generated if forwarding fails (not upfront)
        control_bytes = local_ack.total_size_bytes

        self.action_stats[AckAction.A3_LOCAL_SELECTIVE]['control_packets'] += 1
        energy = config.get('energy_cost_mj', 0.5)
        self.action_stats[AckAction.A3_LOCAL_SELECTIVE]['energy_mj'] += energy

        return AckOutcome(
            action=AckAction.A3_LOCAL_SELECTIVE,
            original_packet=packet,
            control_packets_generated=[local_ack],
            total_control_bytes=control_bytes,
            total_control_packets=2,  # Local ACK + potential NACKs
            handshake_delay_ms=config.get('handshake_delay_ms', 3.0),
            energy_cost_mj=energy,
            retransmission_limit=retransmit_limit,
            requires_end_to_end_ack=False,
            requires_local_ack=True,
            requires_forwarding=True,
            description="Local ACK + Selective Retransmit: local ACK, "
                        "edge router retransmits to failed subscribers (up to 4 retries)",
        )

    def _handle_a4(self, packet: Packet, sim_time: float) -> AckOutcome:
        """
        a4: QoS 2-like Multi-step Handshake.

        Four-step handshake for exactly-once delivery guarantee:
        1. Publisher -> PUBLISH -> Broker
        2. Broker -> PUBREC -> Publisher
        3. Publisher -> PUBREL -> Broker
        4. Broker -> PUBCOMP -> Publisher

        Each step can be retransmitted (up to 5 retries total).
        Highest reliability but also highest overhead and delay.

        Packet flow:
          Publisher -> Broker (PUBLISH)
          Broker -> Publisher (PUBREC)
          Publisher -> Broker (PUBREL)
          Broker -> Publisher (PUBCOMP)
        """
        config = self.config.get('a4', {})
        retransmit_limit = config.get('retransmission_limit', 5)
        packet.max_retransmits = retransmit_limit

        # Generate all handshake packets
        pubrec = create_ack_packet(packet, PacketType.PUBREC, sim_time)
        pubrel = create_ack_packet(packet, PacketType.PUBREL, sim_time)
        pubcomp = create_ack_packet(packet, PacketType.PUBCOMP, sim_time)

        control_packets = [pubrec, pubrel, pubcomp]
        total_bytes = sum(p.total_size_bytes for p in control_packets)

        self.action_stats[AckAction.A4_QOS2_HANDSHAKE]['control_packets'] += 3
        energy = config.get('energy_cost_mj', 0.8)
        self.action_stats[AckAction.A4_QOS2_HANDSHAKE]['energy_mj'] += energy

        return AckOutcome(
            action=AckAction.A4_QOS2_HANDSHAKE,
            original_packet=packet,
            control_packets_generated=control_packets,
            total_control_bytes=total_bytes,
            total_control_packets=4,  # DATA + 3 control = 4 packets total
            handshake_delay_ms=config.get('handshake_delay_ms', 5.0),
            energy_cost_mj=energy,
            retransmission_limit=retransmit_limit,
            requires_end_to_end_ack=True,
            requires_local_ack=False,
            requires_forwarding=False,
            description="QoS 2 handshake: 4-step exactly-once delivery (up to 5 retries)",
        )

    def get_action_overhead_bytes(self, action: AckAction) -> int:
        """Get the control overhead in bytes for an action (no retransmits)."""
        return MAX_OVERHEAD_BYTES.get(action, 0)

    def get_stats(self) -> dict:
        """Get per-action statistics."""
        return {
            action.name: {
                'applied': stats['applied'],
                'control_packets': stats['control_packets'],
                'retransmits': stats['retransmits'],
                'energy_mj': round(stats['energy_mj'], 4),
            }
            for action, stats in self.action_stats.items()
        }

    def reset_stats(self) -> None:
        """Reset all action statistics."""
        for stats in self.action_stats.values():
            stats['applied'] = 0
            stats['control_packets'] = 0
            stats['retransmits'] = 0
            stats['energy_mj'] = 0.0
