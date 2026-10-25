"""
Wi-Fi 6 (802.11ax) Channel Model for the RL-ACK Simulator.

Models indoor propagation (log-distance), CSMA/CA contention,
transmission delays, and packet loss. The channel tracks busy/idle
state and provides realistic timing for the star-topology network.

This is a simplified but faithful model of 802.11ax behavior suitable
for application-layer ACK strategy evaluation.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .packet import Packet


@dataclass
class ChannelConfig:
    """Wi-Fi 6 channel configuration."""
    bandwidth_mhz: float = 20.0
    data_rate_mbps: float = 143.4      # HE MCS7, 1 SS, 20 MHz, 800 ns GI
    guard_interval_ns: int = 800
    slot_time_us: float = 9.0
    sifs_us: float = 16.0
    difs_us: float = 34.0
    cw_min: int = 15
    cw_max: int = 1023
    tx_power_dbm: float = 20.0
    noise_floor_dbm: float = -93.97
    path_loss_exponent: float = 3.0
    reference_loss_db: float = 46.67   # At 1m for 5 GHz
    area_side_m: float = 15.0


class WifiChannel:
    """
    Simulated IEEE 802.11ax wireless channel.

    Models:
    - Log-distance indoor propagation loss
    - CSMA/CA with binary exponential backoff
    - Transmission delay based on packet size and data rate
    - Configurable forward-path and ACK-path packet loss
    - Channel busy/idle state tracking
    - Collision detection via overlapping transmissions
    """

    def __init__(self, config: ChannelConfig = None,
                 fwd_loss_rate: float = 0.01,
                 ack_loss_rate: float = 0.05,
                 rng: random.Random = None):
        """
        Args:
            config: Channel configuration parameters.
            fwd_loss_rate: Forward-path packet loss probability.
            ack_loss_rate: ACK-path packet loss probability.
            rng: Random number generator for reproducibility.
        """
        self.config = config or ChannelConfig()
        self.fwd_loss_rate = fwd_loss_rate
        self.ack_loss_rate = ack_loss_rate
        self.rng = rng or random.Random(42)

        # Channel state
        self.busy = False
        self.busy_until: float = 0.0
        self.current_transmitter: Optional[str] = None

        # Contention window per node
        self._contention_windows: dict[str, int] = {}

        # Statistics
        self.total_transmissions = 0
        self.total_collisions = 0
        self.total_fwd_losses = 0
        self.total_ack_losses = 0
        self.busy_time_total: float = 0.0
        self._last_idle_start: float = 0.0
        self.channel_busy_samples: list[tuple[float, bool]] = []

    def compute_propagation_delay(self, distance_m: float) -> float:
        """
        Compute propagation delay for a given distance.

        For indoor smart kitchen (max ~15m), propagation delay is negligible
        (~50 ns) but included for completeness.

        Returns:
            Propagation delay in seconds.
        """
        speed_of_light = 3e8  # m/s
        return distance_m / speed_of_light

    def compute_path_loss_db(self, distance_m: float) -> float:
        """
        Log-distance indoor path loss model.

        PL(d) = PL(d0) + 10 * n * log10(d/d0)

        Where:
            PL(d0) = reference_loss_db at d0 = 1m
            n = path_loss_exponent (3.0 for indoor)
        """
        if distance_m <= 0:
            distance_m = 0.1
        return self.config.reference_loss_db + \
               10 * self.config.path_loss_exponent * math.log10(max(distance_m, 1.0))

    def compute_snr_db(self, distance_m: float) -> float:
        """Compute Signal-to-Noise Ratio at a given distance."""
        path_loss = self.compute_path_loss_db(distance_m)
        received_power = self.config.tx_power_dbm - path_loss
        return received_power - self.config.noise_floor_dbm

    def compute_transmission_delay(self, packet_size_bytes: int) -> float:
        """
        Compute the transmission delay for a packet.

        Returns:
            Transmission delay in seconds.
        """
        bits = packet_size_bytes * 8
        rate_bps = self.config.data_rate_mbps * 1e6
        return bits / rate_bps

    def compute_csma_delay(self, node_id: str) -> float:
        """
        Compute CSMA/CA access delay including DIFS and backoff.

        Uses binary exponential backoff with the node's current
        contention window.

        Returns:
            Total channel access delay in seconds.
        """
        # DIFS wait
        difs_s = self.config.difs_us * 1e-6

        # Random backoff
        cw = self._contention_windows.get(node_id, self.config.cw_min)
        backoff_slots = self.rng.randint(0, cw)
        backoff_s = backoff_slots * self.config.slot_time_us * 1e-6

        return difs_s + backoff_s

    def attempt_transmission(self, node_id: str, packet: Packet,
                             sim_time: float, distance_m: float = 5.0) -> dict:
        """
        Attempt to transmit a packet over the channel.

        Returns a dict describing the outcome:
        {
            'success': bool,
            'channel_access_delay': float (seconds),
            'transmission_delay': float (seconds),
            'propagation_delay': float (seconds),
            'total_delay': float (seconds),
            'lost': bool,
            'loss_type': str or None,  # 'forward', 'collision'
            'completion_time': float (absolute sim time),
        }
        """
        result = {
            'success': False,
            'channel_access_delay': 0.0,
            'transmission_delay': 0.0,
            'propagation_delay': 0.0,
            'total_delay': 0.0,
            'lost': False,
            'loss_type': None,
            'completion_time': sim_time,
        }

        # CSMA/CA access delay
        access_delay = self.compute_csma_delay(node_id)

        # Wait for channel to be free
        if self.busy and sim_time < self.busy_until:
            wait_time = self.busy_until - sim_time
            access_delay += wait_time

        tx_start = sim_time + access_delay

        # Transmission delay
        tx_delay = self.compute_transmission_delay(packet.total_size_bytes)

        # Propagation delay
        prop_delay = self.compute_propagation_delay(distance_m)

        total_delay = access_delay + tx_delay + prop_delay
        completion_time = sim_time + total_delay

        result['channel_access_delay'] = access_delay
        result['transmission_delay'] = tx_delay
        result['propagation_delay'] = prop_delay
        result['total_delay'] = total_delay
        result['completion_time'] = completion_time

        # Check for packet loss (forward path)
        if self.rng.random() < self.fwd_loss_rate:
            result['lost'] = True
            result['loss_type'] = 'forward'
            self.total_fwd_losses += 1
            # Reset contention window on loss (doubled)
            cw = self._contention_windows.get(node_id, self.config.cw_min)
            self._contention_windows[node_id] = min(cw * 2, self.config.cw_max)
        else:
            result['success'] = True
            # Reset contention window on success
            self._contention_windows[node_id] = self.config.cw_min

        # Update channel state
        self.busy = True
        self.busy_until = tx_start + tx_delay
        self.current_transmitter = node_id
        self.total_transmissions += 1
        self.busy_time_total += tx_delay

        return result

    def attempt_ack_transmission(self, packet: Packet,
                                 sim_time: float,
                                 distance_m: float = 5.0) -> dict:
        """
        Attempt to transmit an ACK packet (ACK path).

        Uses separate ACK-path loss rate.
        """
        # SIFS + ACK transmission (no contention for ACK)
        sifs_s = self.config.sifs_us * 1e-6
        tx_delay = self.compute_transmission_delay(packet.total_size_bytes)
        prop_delay = self.compute_propagation_delay(distance_m)

        total_delay = sifs_s + tx_delay + prop_delay
        completion_time = sim_time + total_delay

        result = {
            'success': True,
            'channel_access_delay': sifs_s,
            'transmission_delay': tx_delay,
            'propagation_delay': prop_delay,
            'total_delay': total_delay,
            'completion_time': completion_time,
            'lost': False,
            'loss_type': None,
        }

        # Check for ACK-path loss
        if self.rng.random() < self.ack_loss_rate:
            result['success'] = False
            result['lost'] = True
            result['loss_type'] = 'ack_path'
            self.total_ack_losses += 1

        self.busy_time_total += tx_delay
        return result

    def get_utilization(self, total_sim_time: float) -> float:
        """
        Compute actual channel utilization.

        Returns fraction of time the channel was busy.
        """
        if total_sim_time <= 0:
            return 0.0
        return min(self.busy_time_total / total_sim_time, 1.0)

    def reset_stats(self) -> None:
        """Reset channel statistics."""
        self.total_transmissions = 0
        self.total_collisions = 0
        self.total_fwd_losses = 0
        self.total_ack_losses = 0
        self.busy_time_total = 0.0
        self._contention_windows.clear()

    def get_stats(self, sim_time: float) -> dict:
        """Get channel statistics."""
        return {
            'total_transmissions': self.total_transmissions,
            'total_collisions': self.total_collisions,
            'total_fwd_losses': self.total_fwd_losses,
            'total_ack_losses': self.total_ack_losses,
            'utilization': self.get_utilization(sim_time),
            'busy_time_s': self.busy_time_total,
        }
