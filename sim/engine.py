"""
Discrete Event Simulation (DES) Engine for RL-ACK Smart Kitchen IoT Simulator.

This module implements a priority-queue-based event scheduler that drives
the entire simulation. All network behaviors (transmission, reception,
loss, timeout, ACK) are modeled as discrete events with precise timestamps.

This is NOT an analytical model — every packet goes through actual
queuing, channel contention, transmission, and reception events.
"""

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class EventType(Enum):
    """All discrete event types in the simulation."""
    # Sensor events
    SENSOR_READING = auto()
    SENSOR_EVENT_START = auto()
    SENSOR_EVENT_END = auto()
    SENSOR_FAILURE = auto()
    SENSOR_RECOVERY = auto()

    # Message lifecycle
    MESSAGE_GENERATE = auto()
    MESSAGE_ENQUEUE = auto()
    MESSAGE_DEQUEUE = auto()

    # Channel/transmission events
    CHANNEL_SENSE = auto()
    BACKOFF_START = auto()
    BACKOFF_END = auto()
    TRANSMIT_START = auto()
    TRANSMIT_END = auto()
    PACKET_ARRIVE = auto()
    PACKET_LOSS = auto()

    # ACK events
    ACK_DECIDE = auto()
    ACK_SEND = auto()
    ACK_ARRIVE = auto()
    ACK_LOSS = auto()
    ACK_TIMEOUT = auto()
    RETRANSMIT = auto()

    # Handshake events (for a3, a4)
    HANDSHAKE_STEP = auto()
    HANDSHAKE_COMPLETE = auto()
    LOCAL_ACK_SEND = auto()
    LOCAL_ACK_ARRIVE = auto()
    FORWARD_START = auto()
    FORWARD_COMPLETE = auto()
    SELECTIVE_RETRANSMIT = auto()

    # Delivery outcomes
    DELIVERY_SUCCESS = auto()
    DELIVERY_FAILURE = auto()
    DEADLINE_EXPIRED = auto()

    # RL events
    DQN_STEP = auto()
    DQN_TRAIN = auto()

    # Monitoring
    METRIC_SAMPLE = auto()
    QUEUE_CHECK = auto()
    UTILIZATION_SAMPLE = auto()

    # Simulation control
    SIM_START = auto()
    SIM_END = auto()


@dataclass(order=True)
class Event:
    """
    A single simulation event.

    Events are ordered by timestamp (earliest first), then by priority
    (lower number = higher priority), then by insertion order.
    """
    timestamp: float  # Simulation time in seconds
    priority: int = field(compare=True, default=0)
    sequence: int = field(compare=True, default=0)
    event_type: EventType = field(compare=False, default=EventType.SIM_START)
    data: dict = field(compare=False, default_factory=dict)
    callback: Optional[Callable] = field(compare=False, default=None, repr=False)
    source_id: str = field(compare=False, default="")
    description: str = field(compare=False, default="")


class SimulationEngine:
    """
    Priority-queue-based Discrete Event Simulation engine.

    Manages a global simulation clock and an event queue. Handlers are
    registered for each EventType and called when events are dequeued.

    Usage:
        engine = SimulationEngine(duration=300.0, seed=42)
        engine.register_handler(EventType.TRANSMIT_START, my_handler)
        engine.schedule(0.0, EventType.SIM_START, {})
        engine.run()
    """

    def __init__(self, duration: float = 300.0, seed: int = 42):
        """
        Args:
            duration: Total simulation time in seconds.
            seed: Random seed for reproducibility.
        """
        self.duration = duration
        self.seed = seed
        self.clock: float = 0.0  # Current simulation time (seconds)
        self._event_queue: list[Event] = []
        self._sequence_counter: int = 0
        self._handlers: dict[EventType, list[Callable]] = {}
        self._running: bool = False
        self._paused: bool = False
        self._event_count: int = 0
        self._events_processed: int = 0
        self._wall_start: float = 0.0

        # Statistics
        self.stats = {
            'total_events_scheduled': 0,
            'total_events_processed': 0,
            'events_by_type': {},
            'wall_clock_time_s': 0.0,
            'sim_time_reached': 0.0,
        }

    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register a callback for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def schedule(self, timestamp: float, event_type: EventType,
                 data: dict = None, priority: int = 0,
                 callback: Optional[Callable] = None,
                 source_id: str = "", description: str = "") -> Event:
        """
        Schedule an event at a future simulation time.

        Args:
            timestamp: When the event should fire (absolute sim time in seconds).
            event_type: The type of event.
            data: Arbitrary data dict attached to the event.
            priority: Lower = higher priority (for tie-breaking).
            callback: Optional direct callback (in addition to registered handlers).
            source_id: Identifier of the entity scheduling this event.
            description: Human-readable description.

        Returns:
            The scheduled Event object.

        Raises:
            ValueError: If timestamp is in the past.
        """
        if timestamp < self.clock - 1e-12:
            raise ValueError(
                f"Cannot schedule event in the past: "
                f"event_time={timestamp:.6f}, clock={self.clock:.6f}"
            )

        self._sequence_counter += 1
        event = Event(
            timestamp=timestamp,
            priority=priority,
            sequence=self._sequence_counter,
            event_type=event_type,
            data=data or {},
            callback=callback,
            source_id=source_id,
            description=description,
        )
        heapq.heappush(self._event_queue, event)
        self.stats['total_events_scheduled'] += 1
        return event

    def schedule_relative(self, delay: float, event_type: EventType,
                          data: dict = None, **kwargs) -> Event:
        """Schedule an event relative to the current simulation time."""
        return self.schedule(self.clock + delay, event_type, data, **kwargs)

    def cancel_event(self, event: Event) -> bool:
        """
        Cancel a scheduled event by marking it.

        Note: Due to heap structure, we mark events as cancelled rather
        than removing them. They are skipped during processing.
        """
        event.event_type = None  # type: ignore
        return True

    def run(self, progress_callback: Optional[Callable] = None,
            progress_interval: float = 10.0) -> dict:
        """
        Run the simulation until completion or duration exceeded.

        Args:
            progress_callback: Called periodically with (clock, duration, events_processed).
            progress_interval: How often to call progress_callback (sim seconds).

        Returns:
            Simulation statistics dict.
        """
        self._running = True
        self._wall_start = time.time()
        next_progress = progress_interval

        while self._event_queue and self._running:
            event = heapq.heappop(self._event_queue)

            # Skip cancelled events
            if event.event_type is None:
                continue

            # Stop if we've exceeded simulation duration
            if event.timestamp > self.duration:
                break

            # Advance clock
            self.clock = event.timestamp
            self._events_processed += 1

            # Track event type statistics
            etype_name = event.event_type.name
            self.stats['events_by_type'][etype_name] = \
                self.stats['events_by_type'].get(etype_name, 0) + 1

            # Execute direct callback if provided
            if event.callback is not None:
                try:
                    event.callback(event)
                except Exception as e:
                    print(f"[DES] Error in callback for {event.event_type}: {e}")

            # Execute registered handlers
            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    print(f"[DES] Error in handler for {event.event_type}: {e}")

            # Progress reporting
            if progress_callback and self.clock >= next_progress:
                progress_callback(self.clock, self.duration, self._events_processed)
                next_progress += progress_interval

        wall_elapsed = time.time() - self._wall_start
        self.stats['total_events_processed'] = self._events_processed
        self.stats['wall_clock_time_s'] = wall_elapsed
        self.stats['sim_time_reached'] = self.clock
        self._running = False
        return self.stats

    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.clock = 0.0
        self._event_queue.clear()
        self._sequence_counter = 0
        self._events_processed = 0
        self._running = False
        self._paused = False
        self.stats = {
            'total_events_scheduled': 0,
            'total_events_processed': 0,
            'events_by_type': {},
            'wall_clock_time_s': 0.0,
            'sim_time_reached': 0.0,
        }

    @property
    def pending_events(self) -> int:
        """Number of events still in the queue."""
        return len(self._event_queue)

    @property
    def is_running(self) -> bool:
        return self._running

    def peek_next(self) -> Optional[Event]:
        """Peek at the next event without removing it."""
        if self._event_queue:
            return self._event_queue[0]
        return None
