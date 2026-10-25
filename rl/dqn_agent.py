"""
Deep Q-Network (DQN) Agent for the RL-ACK Framework.

Network architecture (from paper):
  Input:  8 normalized features
  Hidden: 64 neurons (ReLU) -> 32 neurons (ReLU)
  Output: 5 Q-values (one per ACK action)

Features:
  Adam optimizer, experience replay, target network,
  epsilon-greedy exploration, Bellman loss, checkpointing.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """
    DQN neural network.

    Architecture matches the paper specification:
      Input(8) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(5)
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 5,
                 hidden_1: int = 64, hidden_2: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> Q-values."""
        return self.network(x)


class DQNAgent:
    """
    DQN Agent for adaptive ACK selection.

    Implements:
    - Epsilon-greedy exploration with decay
    - Experience replay (uniform sampling)
    - Target network with periodic updates
    - Bellman loss (MSE)
    - Model checkpointing
    - Deterministic evaluation mode
    """

    def __init__(self, config: dict, device: str = None, seed: int = None):
        """
        Args:
            config: DQN configuration from default_config.json['dqn'].
            device: PyTorch device ('cpu' or 'cuda').
            seed: Optional random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.state_dim = config.get('state_dim', 8)
        self.action_dim = config.get('action_dim', 5)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.policy_net = DQNNetwork(
            self.state_dim, self.action_dim,
            config.get('hidden_1', 64),
            config.get('hidden_2', 32),
        ).to(self.device)

        self.target_net = DQNNetwork(
            self.state_dim, self.action_dim,
            config.get('hidden_1', 64),
            config.get('hidden_2', 32),
        ).to(self.device)

        # Initialize target network with policy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 0.001),
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('replay_buffer_size', 10000),
        )

        # Hyperparameters
        self.gamma = config.get('discount_factor', 0.99)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_interval = config.get('target_update_interval', 100)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 50000)
        self.training_steps = config.get('training_steps', 100000)
        self.checkpoint_interval = config.get('checkpoint_interval', 5000)

        # State
        self.step_count = 0
        self.train_count = 0
        self.total_loss = 0.0
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []
        self.action_history: list[int] = []
        self.epsilon_history: list[float] = []
        self.is_training = True

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: 8-dimensional normalized state vector.
            deterministic: If True, always select greedy action (evaluation mode).

        Returns:
            Action index (0-4).
        """
        self.step_count += 1

        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()

        self.action_history.append(action)

        # Decay epsilon
        if self.is_training:
            self._decay_epsilon()

        return action

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.reward_history.append(reward)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (sample batch, compute loss, update weights).

        Returns:
            Loss value, or None if buffer not ready.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions taken
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Bellman loss (MSE)
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.total_loss += loss_val
        self.train_count += 1
        self.loss_history.append(loss_val)

        # Update target network periodically
        if self.train_count % self.target_update_interval == 0:
            self.update_target_network()

        return loss_val

    def update_target_network(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _decay_epsilon(self) -> None:
        """Linear epsilon decay."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay_steps
        )
        self.epsilon_history.append(self.epsilon)

    def set_eval_mode(self) -> None:
        """Set agent to evaluation mode (deterministic actions, no training)."""
        self.is_training = False
        self.policy_net.eval()

    def set_train_mode(self) -> None:
        """Set agent to training mode."""
        self.is_training = True
        self.policy_net.train()

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'train_count': self.train_count,
            'epsilon': self.epsilon,
            'total_loss': self.total_loss,
            'config': self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        self.train_count = checkpoint.get('train_count', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.total_loss = checkpoint.get('total_loss', 0.0)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]

    def get_stats(self) -> dict:
        """Get training statistics."""
        recent_rewards = self.reward_history[-100:] if self.reward_history else []
        recent_losses = self.loss_history[-100:] if self.loss_history else []

        return {
            'step_count': self.step_count,
            'train_count': self.train_count,
            'epsilon': round(self.epsilon, 4),
            'buffer_size': len(self.replay_buffer),
            'avg_reward_recent': float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            'avg_loss_recent': float(np.mean(recent_losses)) if recent_losses else 0.0,
            'total_loss': self.total_loss,
            'action_distribution': {
                i: self.action_history.count(i)
                for i in range(self.action_dim)
            } if self.action_history else {},
        }
