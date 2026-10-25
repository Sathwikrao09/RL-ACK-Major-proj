"""
RL-ACK Controller Interface for Edge Router.

Defines the interface between the network edge router and the ACK decision logic.
Supports:
  1. Direct in-memory invocation (fast discrete-event simulation)
  2. Socket/IPC invocation (for NS-3 C++ or distributed deployment)
  3. JSON file-based exchange (cross-platform fallback)
"""

import json
import socket
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ack.actions import AckAction
from sim.packet import Packet


class ControllerInterface(ABC):
    """Abstract interface for the RL-ACK decision controller."""

    @abstractmethod
    def select_action(self, state: np.ndarray, packet: Packet) -> AckAction:
        """Select an ACK action given state vector and packet metadata."""
        pass

    @abstractmethod
    def feedback(self, state: np.ndarray, action: AckAction, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Provide transition feedback for training."""
        pass


class LocalController(ControllerInterface):
    """In-process controller using a Python Policy or DQNAgent directly."""

    def __init__(self, policy, dqn_agent=None):
        self.policy = policy
        self.dqn_agent = dqn_agent

    def select_action(self, state: np.ndarray, packet: Packet) -> AckAction:
        return self.policy.select_action(state, packet)

    def feedback(self, state: np.ndarray, action: AckAction, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        if self.dqn_agent and self.dqn_agent.is_training:
            self.dqn_agent.store_transition(state, action.value, reward, next_state, done)
            self.dqn_agent.train_step()


class SocketControllerServer:
    """
    TCP Socket Server providing ACK decisions to external simulators (like NS-3 C++).

    Protocol:
      Request:  JSON line with {"type": "decide", "state": [...], "packet_id": "...", "tier": "CC"}
      Response: JSON line with {"action": 0..4, "action_name": "aX"}
      Feedback: JSON line with {"type": "feedback", "state": [...], "action": 2, "reward": 0.5, ...}
      Ack:      JSON line with {"status": "ok"}
    """

    def __init__(self, controller: ControllerInterface, host: str = "127.0.0.1", port: int = 5555):
        self.controller = controller
        self.host = host
        self.port = port
        self.server_sock = None
        self.running = False

    def start(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        self.running = True
        print(f"[SocketController] Listening on {self.host}:{self.port}")

    def handle_client(self, conn):
        with conn:
            buffer = ""
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    req_type = req.get("type", "decide")

                    if req_type == "decide":
                        state = np.array(req.get("state", [0.0]*8), dtype=np.float32)
                        action = self.controller.select_action(state, None)
                        resp = {"action": action.value, "action_name": action.name}
                        conn.sendall((json.dumps(resp) + "\n").encode('utf-8'))

                    elif req_type == "feedback":
                        state = np.array(req.get("state", [0.0]*8), dtype=np.float32)
                        action = AckAction(req.get("action", 0))
                        reward = float(req.get("reward", 0.0))
                        next_state = np.array(req.get("next_state", [0.0]*8), dtype=np.float32)
                        done = bool(req.get("done", False))
                        self.controller.feedback(state, action, reward, next_state, done)
                        conn.sendall(b'{"status":"ok"}\n')

    def stop(self):
        self.running = False
        if self.server_sock:
            self.server_sock.close()
