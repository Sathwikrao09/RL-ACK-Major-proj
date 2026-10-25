"""
Baseline Policies for the RL-ACK Framework Comparison.

Six policies:
  1. Static QoS 0-like (always a0)
  2. Static Local ACK (always a1)
  3. Static QoS 1-like (always a2)
  4. Static Local ACK + Selective Retransmission (always a3)
  5. Static QoS 2-like (always a4)
  6. Adaptive RL-ACK DQN (learned policy)

All policies use identical network settings, traffic, and random seeds.
"""

from typing import Optional
import numpy as np

from ack.actions import AckAction
from sim.packet import Packet, TrafficTier


class BasePolicy:
    """Base class for ACK selection policies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.decisions_made = 0

    def select_action(self, state: np.ndarray = None,
                      packet: Packet = None) -> AckAction:
        """Select an ACK action. Override in subclasses."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset policy state."""
        self.decisions_made = 0


class StaticPolicy(BasePolicy):
    """Always selects the same fixed ACK action."""

    def __init__(self, action: AckAction):
        names = {
            AckAction.A0_NO_ACK: ("Static QoS 0", "Always no ACK (fire-and-forget)"),
            AckAction.A1_LOCAL_ACK: ("Static Local ACK", "Always local/proxy ACK"),
            AckAction.A2_PUBACK: ("Static QoS 1", "Always PUBACK with retransmission"),
            AckAction.A3_LOCAL_SELECTIVE: ("Static Local+Selective",
                                           "Always local ACK with selective retransmission"),
            AckAction.A4_QOS2_HANDSHAKE: ("Static QoS 2", "Always QoS 2 multi-step handshake"),
        }
        name, desc = names.get(action, (f"Static {action.name}", "Fixed action"))
        super().__init__(name, desc)
        self.fixed_action = action

    def select_action(self, state: np.ndarray = None,
                      packet: Packet = None) -> AckAction:
        self.decisions_made += 1
        return self.fixed_action


class DQNPolicy(BasePolicy):
    """Uses the trained DQN agent to select actions."""

    def __init__(self, agent, deterministic: bool = True):
        super().__init__("RL-ACK DQN", "Adaptive DQN-based ACK selection")
        self.agent = agent
        self.deterministic = deterministic

    def select_action(self, state: np.ndarray = None,
                      packet: Packet = None) -> AckAction:
        self.decisions_made += 1
        if state is None:
            return AckAction.A2_PUBACK  # Default fallback
        action_idx = self.agent.select_action(state, deterministic=self.deterministic)
        return AckAction(action_idx)


def get_all_policies(dqn_agent=None) -> list[BasePolicy]:
    """
    Get all six policies for comparison.

    Args:
        dqn_agent: Trained DQN agent (or None to skip RL-ACK).

    Returns:
        List of all six policies.
    """
    policies = [
        StaticPolicy(AckAction.A0_NO_ACK),
        StaticPolicy(AckAction.A1_LOCAL_ACK),
        StaticPolicy(AckAction.A2_PUBACK),
        StaticPolicy(AckAction.A3_LOCAL_SELECTIVE),
        StaticPolicy(AckAction.A4_QOS2_HANDSHAKE),
    ]
    if dqn_agent is not None:
        policies.append(DQNPolicy(dqn_agent, deterministic=True))
    return policies


def get_policy_by_name(name: str, dqn_agent=None) -> Optional[BasePolicy]:
    """Get a single policy by name."""
    for policy in get_all_policies(dqn_agent):
        if policy.name == name or policy.name.lower().replace(' ', '_') == name.lower():
            return policy
    return None
