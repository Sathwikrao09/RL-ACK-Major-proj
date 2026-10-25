"""
Network Simulation Orchestrator for the RL-ACK Smart Kitchen Simulator.

This is the central module that ties together:
  - Discrete Event Simulation engine
  - Wi-Fi 6 channel model
  - Network nodes (IoT devices + edge router)
  - Smart kitchen sensors with events
  - Traffic generation and classification
  - Five ACK actions with distinct behaviors
  - DQN agent (or static policy)
  - Reward computation
  - Metrics collection

The simulation follows a star topology: all IoT devices communicate
with a central edge router that hosts the RL-ACK controller.
"""

import json
import os
import random
import math
import numpy as np
from typing import Optional
from collections import deque

from .engine import SimulationEngine, EventType, Event
from .channel import WifiChannel, ChannelConfig
from .node import NetworkNode, NodeRole, DeviceCategory, TransmitQueue
from .packet import (
    Packet, PacketType, TrafficTier, DeliveryStatus,
    create_data_packet, create_ack_packet, create_retransmit_packet,
    TIER_DEADLINES
)

from sensors.sensor_models import VirtualSensor, SensorFactory, SensorType
from sensors.event_generator import EventGenerator
from traffic.generator import TrafficGenerator, TrafficConfig
from ack.actions import AckActionHandler, AckAction, AckOutcome, GLOBAL_MAX_OVERHEAD
from rl.dqn_agent import DQNAgent
from rl.reward import RewardCalculator
from metrics.collector import MetricsCollector
from policies.baselines import BasePolicy, StaticPolicy, DQNPolicy


class SmartKitchenSimulation:
    """
    Complete RL-ACK Smart Kitchen IoT network simulation.

    Orchestrates all subsystems through discrete events.
    """

    def __init__(self, config: dict, policy: BasePolicy = None,
                 dqn_agent: DQNAgent = None, seed: int = 42):
        """
        Args:
            config: Full configuration dict.
            policy: ACK selection policy (static or DQN).
            dqn_agent: DQN agent for training (used when policy is DQNPolicy).
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Core parameters
        sim_cfg = config.get('simulation', {})
        net_cfg = config.get('network', {})
        self.duration = sim_cfg.get('duration_s', 300)
        self.num_devices = net_cfg.get('default_device_count', 10)
        self.utilization_target = net_cfg.get('default_utilization', 0.7)
        self.ack_loss_rate = net_cfg.get('default_ack_loss', 0.05)
        self.fwd_loss_rate = net_cfg.get('default_fwd_loss', 0.01)

        # Subsystems
        self.engine = SimulationEngine(duration=self.duration, seed=seed)

        self.channel = WifiChannel(
            config=ChannelConfig(
                data_rate_mbps=net_cfg.get('data_rate_mbps', 143.4),
                area_side_m=net_cfg.get('area_side_m', 15.0),
            ),
            fwd_loss_rate=self.fwd_loss_rate,
            ack_loss_rate=self.ack_loss_rate,
            rng=random.Random(seed + 1),
        )

        self.traffic_gen = TrafficGenerator(
            config.get('traffic', {}),
            rng=random.Random(seed + 2),
        )

        self.ack_handler = AckActionHandler(config.get('ack_actions', {}))

        self.reward_calc = RewardCalculator(config.get('reward', {}))

        self.metrics = MetricsCollector()

        self.event_gen = EventGenerator(
            config.get('sensor_events', {}),
            rng=random.Random(seed + 3),
        )

        # Policy
        self.policy = policy or StaticPolicy(AckAction.A2_PUBACK)
        self.dqn_agent = dqn_agent

        # Nodes
        self.edge_router: Optional[NetworkNode] = None
        self.devices: list[NetworkNode] = []
        self.all_nodes: dict[str, NetworkNode] = {}

        # Sensors
        self.sensors: dict[str, VirtualSensor] = {}

        # State tracking
        self.pending_messages: dict[str, dict] = {}  # message_id -> tracking info
        self.completed_messages: list[dict] = []
        self._rolling_loss_rate = deque(maxlen=100)  # Recent loss outcomes
        self._rolling_latencies = deque(maxlen=100)
        self._recent_overhead = deque(maxlen=100)
        self._total_overhead_bytes = 0
        self._total_energy_mj = 0.0

        # DQN state
        self._last_state = None
        self._last_action = None
        self._dqn_step_count = 0

    def setup(self) -> None:
        """Initialize the network topology, sensors, and schedule initial events."""
        if getattr(self, '_is_setup', False):
            return
        self._is_setup = True
        self._create_topology()
        self._create_sensors()
        self._schedule_initial_events()
        self._register_handlers()

    def _create_topology(self) -> None:
        """Create star topology: edge router at center, devices around it."""
        net_cfg = self.config.get('network', {})
        area_side = net_cfg.get('area_side_m', 15.0)
        center = (area_side / 2, area_side / 2)

        # Create edge router at center
        self.edge_router = NetworkNode(
            node_id="edge_router_0",
            role=NodeRole.EDGE_ROUTER,
            category=DeviceCategory.MIXED,
            position=center,
            max_queue_size=net_cfg.get('max_queue_size', 200),
            rng=random.Random(self.seed + 10),
        )
        self.all_nodes[self.edge_router.node_id] = self.edge_router

        # Determine device categories based on traffic composition
        traffic_cfg = TrafficConfig(self.config.get('traffic', {}))
        tier_counts = traffic_cfg.get_device_counts(self.num_devices)

        # Category mapping
        tier_to_category = {
            TrafficTier.CC: DeviceCategory.CRITICAL_CONTROL,
            TrafficTier.HBS: DeviceCategory.STREAMING,
            TrafficTier.BET: DeviceCategory.TELEMETRY,
        }

        device_idx = 0
        for tier, count in tier_counts.items():
            category = tier_to_category.get(tier, DeviceCategory.TELEMETRY)
            for _ in range(count):
                # Place devices in a circle around the edge router
                angle = 2 * math.pi * device_idx / self.num_devices
                radius = self.rng.uniform(2.0, area_side / 2 - 1.0)
                pos = (
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                )

                node = NetworkNode(
                    node_id=f"device_{device_idx:03d}",
                    role=NodeRole.IOT_DEVICE,
                    category=category,
                    position=pos,
                    max_queue_size=net_cfg.get('max_queue_size', 200),
                    rng=random.Random(self.seed + 100 + device_idx),
                )
                self.devices.append(node)
                self.all_nodes[node.node_id] = node
                device_idx += 1

    def _create_sensors(self) -> None:
        """Create virtual sensors and assign to devices."""
        sensor_configs = self.config.get('sensors', {})

        # Assign sensors based on device category
        category_sensors = {
            DeviceCategory.CRITICAL_CONTROL: ['gas', 'smoke', 'temperature'],
            DeviceCategory.STREAMING: ['temperature', 'humidity'],
            DeviceCategory.TELEMETRY: ['temperature', 'humidity', 'pressure'],
        }

        for device in self.devices:
            sensor_types = category_sensors.get(device.category, ['temperature'])
            for sensor_key in sensor_types:
                if sensor_key in sensor_configs:
                    sensor = SensorFactory.create_sensor(
                        sensor_key, sensor_configs[sensor_key],
                        device_id=device.node_id,
                        rng=random.Random(self.seed + hash(device.node_id + sensor_key) % 10000),
                    )
                    self.sensors[sensor.sensor_id] = sensor
                    device.sensor_ids.append(sensor.sensor_id)

        # Generate events for sensors
        for sensor in self.sensors.values():
            self.event_gen.generate_events_for_sensor(sensor, self.duration)

    def _schedule_initial_events(self) -> None:
        """Schedule the first events to kick off the simulation."""
        # Schedule initial sensor readings for all devices
        for device in self.devices:
            for sensor_id in device.sensor_ids:
                sensor = self.sensors.get(sensor_id)
                if sensor:
                    # Stagger initial readings
                    offset = self.rng.uniform(0, sensor.sampling_interval_s)
                    self.engine.schedule(
                        offset, EventType.SENSOR_READING,
                        {'sensor_id': sensor_id, 'device_id': device.node_id},
                        source_id=device.node_id,
                    )

        # Schedule periodic monitoring
        self.engine.schedule(1.0, EventType.METRIC_SAMPLE, {})

        # Schedule simulation end
        self.engine.schedule(self.duration, EventType.SIM_END, {})

    def _register_handlers(self) -> None:
        """Register event handlers with the DES engine."""
        self.engine.register_handler(EventType.SENSOR_READING, self._on_sensor_reading)
        self.engine.register_handler(EventType.MESSAGE_ENQUEUE, self._on_message_enqueue)
        self.engine.register_handler(EventType.TRANSMIT_START, self._on_transmit_start)
        self.engine.register_handler(EventType.PACKET_ARRIVE, self._on_packet_arrive)
        self.engine.register_handler(EventType.PACKET_LOSS, self._on_packet_loss)
        self.engine.register_handler(EventType.ACK_DECIDE, self._on_ack_decide)
        self.engine.register_handler(EventType.ACK_SEND, self._on_ack_send)
        self.engine.register_handler(EventType.ACK_ARRIVE, self._on_ack_arrive)
        self.engine.register_handler(EventType.ACK_LOSS, self._on_ack_loss)
        self.engine.register_handler(EventType.ACK_TIMEOUT, self._on_ack_timeout)
        self.engine.register_handler(EventType.RETRANSMIT, self._on_retransmit)
        self.engine.register_handler(EventType.DEADLINE_EXPIRED, self._on_deadline_expired)
        self.engine.register_handler(EventType.DELIVERY_SUCCESS, self._on_delivery_success)
        self.engine.register_handler(EventType.METRIC_SAMPLE, self._on_metric_sample)
        self.engine.register_handler(EventType.HANDSHAKE_STEP, self._on_handshake_step)

    # ========================= EVENT HANDLERS =========================

    def _on_sensor_reading(self, event: Event) -> None:
        """Handle a sensor reading event."""
        sensor_id = event.data.get('sensor_id')
        device_id = event.data.get('device_id')
        sensor = self.sensors.get(sensor_id)
        device = self.all_nodes.get(device_id)

        if not sensor or not device:
            return

        # Check for sensor events
        self.event_gen.check_and_trigger_events(self.sensors, self.engine.clock)

        # Generate reading
        reading = sensor.generate_reading(self.engine.clock)

        # Determine traffic tier based on reading
        if reading.alert_level.name in ('CRITICAL', 'EMERGENCY'):
            force_tier = TrafficTier.CC
        elif device.category == DeviceCategory.STREAMING:
            force_tier = TrafficTier.HBS
        elif device.category == DeviceCategory.CRITICAL_CONTROL:
            force_tier = TrafficTier.CC
        else:
            force_tier = TrafficTier.BET

        # Generate message
        packet = self.traffic_gen.generate_message(
            source_node=device,
            dest_id=self.edge_router.node_id,
            sim_time=self.engine.clock,
            sensor_data=reading.to_dict(),
            force_tier=force_tier,
        )

        # Schedule enqueue
        self.engine.schedule(
            self.engine.clock, EventType.MESSAGE_ENQUEUE,
            {'packet': packet, 'device_id': device_id},
            source_id=device_id,
        )

        # Schedule next reading
        interval = sensor.sampling_interval_s
        # Adjust interval by utilization to achieve target load
        if force_tier == TrafficTier.HBS:
            interval = max(0.033, interval * (0.7 / max(0.1, self.utilization_target)))

        self.engine.schedule(
            self.engine.clock + interval, EventType.SENSOR_READING,
            {'sensor_id': sensor_id, 'device_id': device_id},
            source_id=device_id,
        )

    def _on_message_enqueue(self, event: Event) -> None:
        """Enqueue a message at the source device."""
        packet: Packet = event.data.get('packet')
        device_id = event.data.get('device_id')
        device = self.all_nodes.get(device_id)

        if not device or not packet:
            return

        success = device.tx_queue.enqueue(packet, self.engine.clock)
        if not success:
            # Queue full — message dropped
            packet.status = DeliveryStatus.DROPPED
            self.metrics.record_message(packet)
            return

        # Schedule deadline check
        if packet.deadline > 0:
            self.engine.schedule(
                packet.deadline, EventType.DEADLINE_EXPIRED,
                {'message_id': packet.message_id, 'device_id': device_id},
                priority=10,  # Low priority so delivery events fire first
            )

        # If device is not currently transmitting, start transmission
        if not device.is_transmitting:
            self.engine.schedule(
                self.engine.clock, EventType.TRANSMIT_START,
                {'device_id': device_id},
                source_id=device_id,
            )

    def _on_transmit_start(self, event: Event) -> None:
        """Start transmitting the next packet from a device's queue."""
        device_id = event.data.get('device_id')
        device = self.all_nodes.get(device_id)

        if not device or device.tx_queue.is_empty:
            if device:
                device.is_transmitting = False
            return

        packet = device.tx_queue.dequeue(self.engine.clock)
        if not packet:
            device.is_transmitting = False
            return

        device.is_transmitting = True
        packet.transmit_start_time = self.engine.clock
        packet.status = DeliveryStatus.IN_TRANSIT

        # Channel transmission
        distance = device.distance_to(self.edge_router) if self.edge_router else 5.0
        result = self.channel.attempt_transmission(
            device_id, packet, self.engine.clock, distance
        )

        # Energy consumption (modeled)
        energy = device.energy.consume_tx(packet.total_size_bytes)
        packet.energy_consumed_mj += energy

        packet.transmit_end_time = result['completion_time']

        if result['lost']:
            # Forward-path loss
            self.engine.schedule(
                result['completion_time'], EventType.PACKET_LOSS,
                {'packet': packet, 'device_id': device_id, 'loss_type': 'forward'},
                source_id=device_id,
            )
            self._rolling_loss_rate.append(1.0)
        else:
            # Successful transmission -> packet arrives at edge router
            self.engine.schedule(
                result['completion_time'], EventType.PACKET_ARRIVE,
                {'packet': packet, 'device_id': device_id},
                source_id=device_id,
            )
            self._rolling_loss_rate.append(0.0)

        # Schedule next transmission from this device
        self.engine.schedule(
            result['completion_time'] + 0.0001, EventType.TRANSMIT_START,
            {'device_id': device_id},
            source_id=device_id,
        )

    def _on_packet_arrive(self, event: Event) -> None:
        """Handle packet arrival at the edge router."""
        packet: Packet = event.data.get('packet')
        device_id = event.data.get('device_id')

        if not packet:
            return

        packet.arrival_time = self.engine.clock
        self.edge_router.messages_received += 1

        # Consume receive energy at edge router
        self.edge_router.energy.consume_rx(packet.total_size_bytes)

        # Decide ACK action
        self.engine.schedule(
            self.engine.clock, EventType.ACK_DECIDE,
            {'packet': packet, 'device_id': device_id},
            source_id=self.edge_router.node_id,
        )

    def _on_ack_decide(self, event: Event) -> None:
        """RL-ACK controller decides which ACK action to apply."""
        packet: Packet = event.data.get('packet')
        device_id = event.data.get('device_id')
        device = self.all_nodes.get(device_id)

        if not packet or not device:
            return

        # Build state vector for DQN
        state = self._build_state_vector(packet, device)

        # Select action using policy
        if isinstance(self.policy, DQNPolicy) and self.dqn_agent:
            action = self.policy.select_action(state, packet)
        else:
            action = self.policy.select_action(state, packet)

        # Apply ACK action
        outcome = self.ack_handler.apply_action(action, packet, self.engine.clock)

        # Track overhead
        self._total_overhead_bytes += outcome.total_control_bytes
        self._recent_overhead.append(outcome.total_control_bytes)
        self._total_energy_mj += outcome.energy_cost_mj
        packet.energy_consumed_mj += outcome.energy_cost_mj

        # Store pending message info
        self.pending_messages[packet.message_id] = {
            'packet': packet,
            'device_id': device_id,
            'action': action,
            'outcome': outcome,
            'state': state,
            'overhead_bytes': outcome.total_control_bytes,
        }

        # Handle based on action type
        if action == AckAction.A0_NO_ACK:
            # No ACK — mark as delivered immediately
            packet.mark_delivered(self.engine.clock)
            self._finalize_message(packet, outcome, state, device)

        elif action == AckAction.A1_LOCAL_ACK:
            # Send local ACK back to device
            self._schedule_ack_send(outcome.control_packets_generated[0], device)

        elif action == AckAction.A2_PUBACK:
            # Send PUBACK — end-to-end
            self._schedule_ack_send(outcome.control_packets_generated[0], device)

        elif action == AckAction.A3_LOCAL_SELECTIVE:
            # Send local ACK immediately
            self._schedule_ack_send(outcome.control_packets_generated[0], device)
            # Schedule forwarding and selective retransmission monitoring
            self.engine.schedule_relative(
                outcome.handshake_delay_ms / 1000.0,
                EventType.HANDSHAKE_STEP,
                {'message_id': packet.message_id, 'step': 'forward',
                 'retransmit_count': 0},
            )

        elif action == AckAction.A4_QOS2_HANDSHAKE:
            # Start QoS 2 handshake: send PUBREC
            self._schedule_ack_send(outcome.control_packets_generated[0], device)
            # Schedule remaining handshake steps
            self.engine.schedule_relative(
                outcome.handshake_delay_ms / 1000.0 / 3,
                EventType.HANDSHAKE_STEP,
                {'message_id': packet.message_id, 'step': 'pubrel',
                 'packets': outcome.control_packets_generated[1:]},
            )

    def _schedule_ack_send(self, ack_packet: Packet,
                           dest_device: NetworkNode) -> None:
        """Schedule an ACK packet transmission."""
        distance = self.edge_router.distance_to(dest_device)
        result = self.channel.attempt_ack_transmission(
            ack_packet, self.engine.clock, distance
        )

        if result['lost']:
            self.engine.schedule(
                result['completion_time'], EventType.ACK_LOSS,
                {'ack_packet': ack_packet, 'device_id': dest_device.node_id},
            )
        else:
            self.engine.schedule(
                result['completion_time'], EventType.ACK_ARRIVE,
                {'ack_packet': ack_packet, 'device_id': dest_device.node_id},
            )

    def _on_ack_send(self, event: Event) -> None:
        """Handle ACK being sent."""
        pass  # Handled in _schedule_ack_send

    def _on_ack_arrive(self, event: Event) -> None:
        """Handle ACK arrival at the source device."""
        ack_packet: Packet = event.data.get('ack_packet')
        device_id = event.data.get('device_id')
        device = self.all_nodes.get(device_id)

        if not ack_packet or not device:
            return

        message_id = ack_packet.message_id
        pending = self.pending_messages.get(message_id)

        if not pending:
            return

        original_packet = pending['packet']
        original_packet.ack_arrive_time = self.engine.clock
        original_packet.mark_acked(self.engine.clock)
        device.acks_received += 1
        device.record_latency(original_packet.end_to_end_latency or 0.0)

        # Finalize
        self._finalize_message(
            original_packet, pending['outcome'],
            pending['state'], device
        )

    def _on_ack_loss(self, event: Event) -> None:
        """Handle ACK loss — may trigger retransmission."""
        ack_packet: Packet = event.data.get('ack_packet')
        device_id = event.data.get('device_id')

        if not ack_packet:
            return

        message_id = ack_packet.message_id
        pending = self.pending_messages.get(message_id)

        if not pending:
            return

        original = pending['packet']
        action = pending['action']

        # Check retransmission based on action type
        if action in (AckAction.A2_PUBACK, AckAction.A3_LOCAL_SELECTIVE,
                      AckAction.A4_QOS2_HANDSHAKE):
            if original.retransmit_count < original.max_retransmits:
                # Schedule retransmission
                self.engine.schedule_relative(
                    0.01,  # 10ms retransmission delay
                    EventType.RETRANSMIT,
                    {'message_id': message_id, 'device_id': device_id},
                )
            else:
                # Max retransmits reached — delivery failed
                original.mark_timed_out()
                device = self.all_nodes.get(device_id)
                if device:
                    device.messages_failed += 1
                self._finalize_message(
                    original, pending['outcome'],
                    pending['state'], device
                )

    def _on_ack_timeout(self, event: Event) -> None:
        """Handle ACK timeout."""
        message_id = event.data.get('message_id')
        pending = self.pending_messages.get(message_id)
        if pending:
            packet = pending['packet']
            if packet.status not in (DeliveryStatus.ACKED, DeliveryStatus.DELIVERED):
                packet.mark_timed_out()
                device = self.all_nodes.get(pending['device_id'])
                self._finalize_message(
                    packet, pending['outcome'], pending['state'], device
                )

    def _on_retransmit(self, event: Event) -> None:
        """Handle retransmission of a message."""
        message_id = event.data.get('message_id')
        device_id = event.data.get('device_id')
        device = self.all_nodes.get(device_id)
        pending = self.pending_messages.get(message_id)

        if not pending or not device:
            return

        original = pending['packet']
        original.retransmit_count += 1
        self.ack_handler.action_stats[pending['action']]['retransmits'] += 1

        # Create retransmit packet
        retransmit_pkt = create_retransmit_packet(original, self.engine.clock)

        # Re-enqueue
        device.tx_queue.enqueue(retransmit_pkt, self.engine.clock)
        if not device.is_transmitting:
            self.engine.schedule(
                self.engine.clock, EventType.TRANSMIT_START,
                {'device_id': device_id},
            )

    def _on_packet_loss(self, event: Event) -> None:
        """Handle forward-path packet loss."""
        packet: Packet = event.data.get('packet')
        device_id = event.data.get('device_id')
        device = self.all_nodes.get(device_id)

        if not packet:
            return

        # For a0 (no ACK), loss is silent — message just fails
        if packet.ack_action == AckAction.A0_NO_ACK:
            packet.mark_lost()
            if device:
                device.messages_failed += 1
            self.metrics.record_message(packet)
            return

        # For actions with retransmission, check retransmit budget
        if packet.retransmit_count < packet.max_retransmits:
            self.engine.schedule_relative(
                0.01,
                EventType.RETRANSMIT,
                {'message_id': packet.message_id, 'device_id': device_id},
            )
        else:
            packet.mark_lost()
            if device:
                device.messages_failed += 1

            pending = self.pending_messages.get(packet.message_id)
            if pending:
                self._finalize_message(
                    packet, pending.get('outcome'),
                    pending.get('state'), device
                )
            else:
                self.metrics.record_message(packet)

    def _on_deadline_expired(self, event: Event) -> None:
        """Handle message deadline expiration."""
        message_id = event.data.get('message_id')
        pending = self.pending_messages.get(message_id)

        if not pending:
            return

        packet = pending['packet']
        if packet.status in (DeliveryStatus.DELIVERED, DeliveryStatus.ACKED):
            return  # Already delivered

        packet.mark_deadline_missed()
        device = self.all_nodes.get(pending['device_id'])
        if device:
            device.messages_failed += 1
        self._finalize_message(
            packet, pending['outcome'], pending['state'], device
        )

    def _on_delivery_success(self, event: Event) -> None:
        """Handle successful message delivery."""
        pass  # Handled in _finalize_message

    def _on_handshake_step(self, event: Event) -> None:
        """Handle multi-step handshake events (for a3 and a4)."""
        message_id = event.data.get('message_id')
        step = event.data.get('step')
        pending = self.pending_messages.get(message_id)

        if not pending:
            return

        packet = pending['packet']
        device = self.all_nodes.get(pending['device_id'])

        if step == 'forward' and pending['action'] == AckAction.A3_LOCAL_SELECTIVE:
            # a3: Forward to subscriber — check if forwarding succeeds
            retransmit_count = event.data.get('retransmit_count', 0)
            if self.rng.random() < self.fwd_loss_rate and retransmit_count < packet.max_retransmits:
                # Forwarding failed — selective retransmit
                self.engine.schedule_relative(
                    0.005,
                    EventType.HANDSHAKE_STEP,
                    {'message_id': message_id, 'step': 'forward',
                     'retransmit_count': retransmit_count + 1},
                )
                self.ack_handler.action_stats[AckAction.A3_LOCAL_SELECTIVE]['retransmits'] += 1
            else:
                # Forwarding succeeded (or max retries)
                if packet.status not in (DeliveryStatus.ACKED, DeliveryStatus.DELIVERED):
                    packet.mark_delivered(self.engine.clock)
                    self._finalize_message(packet, pending['outcome'],
                                          pending['state'], device)

        elif step == 'pubrel' and pending['action'] == AckAction.A4_QOS2_HANDSHAKE:
            # a4: Continue handshake — PUBREL -> PUBCOMP
            packets = event.data.get('packets', [])
            if packets:
                # Send PUBREL
                self._schedule_ack_send(packets[0], device)
                if len(packets) > 1:
                    self.engine.schedule_relative(
                        0.002,
                        EventType.HANDSHAKE_STEP,
                        {'message_id': message_id, 'step': 'pubcomp',
                         'packets': packets[1:]},
                    )
                else:
                    # Handshake complete
                    if packet.status not in (DeliveryStatus.ACKED, DeliveryStatus.DELIVERED):
                        packet.mark_acked(self.engine.clock)
                        self._finalize_message(packet, pending['outcome'],
                                              pending['state'], device)

            elif step == 'pubcomp':
                if packets:
                    self._schedule_ack_send(packets[0], device)
                if packet.status not in (DeliveryStatus.ACKED, DeliveryStatus.DELIVERED):
                    packet.mark_acked(self.engine.clock)
                    self._finalize_message(packet, pending['outcome'],
                                          pending['state'], device)

    def _on_metric_sample(self, event: Event) -> None:
        """Periodic metric sampling."""
        utilization = self.channel.get_utilization(self.engine.clock)

        # Average queue occupancy across all devices
        avg_queue = 0.0
        if self.devices:
            avg_queue = sum(d.tx_queue.occupancy_fraction for d in self.devices) / len(self.devices)

        # Energy rate
        energy_rate = self._total_energy_mj / max(self.engine.clock, 1.0)

        self.metrics.record_periodic(
            self.engine.clock, utilization, avg_queue, energy_rate
        )

        # Schedule next sample
        if self.engine.clock < self.duration - 1:
            self.engine.schedule(
                self.engine.clock + 1.0, EventType.METRIC_SAMPLE, {}
            )

    # ========================= STATE VECTOR =========================

    def _build_state_vector(self, packet: Packet,
                            device: NetworkNode) -> np.ndarray:
        """
        Build the 8-dimensional normalized state vector for DQN.

        Features:
          0: Network utilization [0,1]
          1: Recent ACK-path packet-loss rate [0,1]
          2: Normalized queue occupancy [0,1]
          3: CC one-hot indicator {0,1}
          4: HBS one-hot indicator {0,1}
          5: BET one-hot indicator {0,1}
          6: Normalized remaining device energy [0,1]
          7: Rolling average latency (normalized by max deadline) [0,1+]
        """
        utilization = self.channel.get_utilization(max(self.engine.clock, 0.001))

        # Recent loss rate
        if self._rolling_loss_rate:
            loss_rate = sum(self._rolling_loss_rate) / len(self._rolling_loss_rate)
        else:
            loss_rate = 0.0

        # Queue occupancy (edge router queue)
        queue_occ = self.edge_router.tx_queue.occupancy_fraction if self.edge_router else 0.0
        # Also consider source device queue
        device_queue = device.tx_queue.occupancy_fraction
        avg_queue = (queue_occ + device_queue) / 2.0

        # Traffic tier one-hot
        cc = 1.0 if packet.traffic_tier == TrafficTier.CC else 0.0
        hbs = 1.0 if packet.traffic_tier == TrafficTier.HBS else 0.0
        bet = 1.0 if packet.traffic_tier == TrafficTier.BET else 0.0

        # Energy remaining
        energy = device.energy.remaining_fraction

        # Rolling average latency (normalized by BET deadline as max)
        max_deadline = 0.5  # BET deadline = 500ms
        rolling_lat = device.rolling_avg_latency / max_deadline if device.rolling_avg_latency > 0 else 0.0

        state = np.array([
            utilization,
            loss_rate,
            avg_queue,
            cc, hbs, bet,
            energy,
            min(rolling_lat, 2.0),  # Cap at 2x for normalization
        ], dtype=np.float32)

        return state

    # ========================= FINALIZATION =========================

    def _finalize_message(self, packet: Packet, outcome: AckOutcome,
                          state: np.ndarray, device: NetworkNode) -> None:
        """Finalize a message: compute reward, train DQN, record metrics."""
        if packet.message_id in self.pending_messages:
            del self.pending_messages[packet.message_id]

        # Record latency
        latency = packet.end_to_end_latency
        if latency is not None:
            self._rolling_latencies.append(latency)
            if device:
                device.record_latency(latency)

        # Compute overhead
        overhead = outcome.total_control_bytes if outcome else 0

        # Record message in metrics
        self.metrics.record_message(
            packet,
            overhead_bytes=overhead,
            energy_mj=packet.energy_consumed_mj,
        )

        if device:
            device.messages_sent += 1

        # DQN training
        if self.dqn_agent and self.dqn_agent.is_training and state is not None:
            action = AckAction(packet.ack_action) if packet.ack_action >= 0 else AckAction.A0_NO_ACK
            utilization = self.channel.get_utilization(max(self.engine.clock, 0.001))

            reward, components = self.reward_calc.compute_reward(
                packet, overhead, utilization, action
            )

            # Build next state
            next_state = self._build_state_vector(packet, device) if device else state

            # Store transition
            self.dqn_agent.store_transition(state, action.value, reward, next_state, False)

            # Train
            loss = self.dqn_agent.train_step()
            self.metrics.record_dqn_step(reward, loss)
            self._dqn_step_count += 1

    # ========================= RUN =========================

    def run(self, progress_callback=None) -> dict:
        """
        Run the full simulation.

        Returns:
            Comprehensive metrics dict.
        """
        self.setup()
        print(f"[SIM] Starting simulation: {self.num_devices} devices, "
              f"duration={self.duration}s, policy={self.policy.name}, "
              f"seed={self.seed}")

        def _progress(clock, duration, events):
            pct = clock / duration * 100
            print(f"  [{pct:5.1f}%] t={clock:.1f}s, events={events}, "
                  f"messages={len(self.metrics.message_records)}")

        stats = self.engine.run(
            progress_callback=progress_callback or _progress,
            progress_interval=self.duration / 10,
        )

        print(f"[SIM] Complete: {stats['total_events_processed']} events processed "
              f"in {stats['wall_clock_time_s']:.1f}s wall time")

        metrics = self.metrics.compute_metrics()
        metrics['engine_stats'] = stats
        metrics['config'] = {
            'num_devices': self.num_devices,
            'duration_s': self.duration,
            'utilization_target': self.utilization_target,
            'ack_loss_rate': self.ack_loss_rate,
            'fwd_loss_rate': self.fwd_loss_rate,
            'policy': self.policy.name,
            'seed': self.seed,
        }

        return metrics

    def get_sensor_states(self) -> list[dict]:
        """Get current state of all sensors (for dashboard)."""
        return [s.get_state() for s in self.sensors.values()]

    def get_sensor_readings(self, limit: int = 100) -> list[dict]:
        """Get recent sensor readings."""
        all_readings = []
        for sensor in self.sensors.values():
            for reading in sensor.reading_history[-limit:]:
                all_readings.append(reading.to_dict())
        all_readings.sort(key=lambda r: r['timestamp'], reverse=True)
        return all_readings[:limit]

    def get_alert_history(self) -> list[dict]:
        """Get all sensor alert history."""
        alerts = []
        for sensor in self.sensors.values():
            alerts.extend(sensor.alert_history)
        alerts.sort(key=lambda a: a['time'], reverse=True)
        return alerts

    def get_node_states(self) -> list[dict]:
        """Get state of all network nodes."""
        return [node.get_state_vector() for node in self.all_nodes.values()]
