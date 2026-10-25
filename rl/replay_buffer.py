"""
Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
provides uniform random sampling for mini-batch training.
"""

import random
from collections import deque
from typing import Optional
import numpy as np


class ReplayBuffer:
    """
    Fixed-size experience replay buffer with uniform sampling.

    Stores transitions as numpy arrays for efficient PyTorch conversion.
    """

    def __init__(self, capacity: int = 10000, seed: int = 42):
        """
        Args:
            capacity: Maximum number of transitions to store.
            seed: Random seed for sampling reproducibility.
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.total_added = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition.

        Args:
            state: Current state vector (8 features).
            action: Action taken (0-4).
            reward: Reward received.
            next_state: Next state vector.
            done: Whether episode ended.
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))
        self.total_added += 1

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        batch = self.rng.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has enough samples for a full batch."""
        return len(self.buffer) >= 64  # Default batch size

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.buffer.clear()
        self.total_added = 0
