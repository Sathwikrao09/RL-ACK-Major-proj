# Research Paper Alignment: RL-ACK Smart Kitchen IoT Framework

This document details the exact alignment between our implementation and the reference paper:
**"RL-ACK: A Reinforcement Learning-Based Adaptive ACK Framework for Heterogeneous Smart Kitchen IoT Networks."**

---

## 1. Network Architecture & Physical Layer

| Feature | Paper Specification | Implementation |
|---|---|---|
| **Topology** | Star topology (IoT nodes to central gateway) | Implemented (`sim.network.SmartKitchenSimulation`) |
| **Physical Standard** | IEEE 802.11ax (Wi-Fi 6) | Implemented (`sim.channel.WifiChannel` & `ns3/scratch/rl-ack-kitchen.cc`) |
| **Channel Bandwidth** | 20 MHz, HE MCS 11 (143.4 Mbps) | Configured in `config/default_config.json` |
| **Area** | Indoor smart kitchen (15m × 15m) | Implemented with Euclidean distance propagation |
| **Gateway Node** | Edge router hosting ACK decision controller | Implemented (`edge_router_0` hosting `DQNAgent`) |

---

## 2. Traffic Classification Tiers

Traffic tiers are assigned **per message**, not permanently per device.

| Tier | Traffic Class | Paper Deadline | Target Share | Arrival Rate | Payload Size | Example Applications |
|---|---|---|---|---|---|---|
| **Tier 1 (CC)** | Critical Control | **10 ms** | 20% | 1–5 msg/s/dev | 64 Bytes | Gas leak alert, emergency valve shutoff, burner flameout |
| **Tier 2 (HBS)** | High-Bandwidth Streaming | **50 ms** | 30% | 15–30 msg/s/dev | 1024 Bytes | Kitchen cameras, thermal vision, pan occupancy streaming |
| **Tier 3 (BET)** | Best-Effort Telemetry | **500 ms** | 50% | 0.1–2 msg/s/dev | 128 Bytes | Routine temperature, humidity, refrigerator compressor status |

---

## 3. The Five ACK Actions

| Action | Paper Definition | Control Packets | Handshake Delay | Retransmit Limit | Modeled Energy |
|---|---|---|---|---|---|
| **a0** | No ACK (QoS 0-like) | 0 | 0.0 ms | 0 | 0.05 mJ |
| **a1** | Local / Proxy ACK | 1 (Edge -> Node) | 1.0 ms | 0 | 0.20 mJ |
| **a2** | PUBACK (QoS 1-like) | 1 (Subscriber -> Node) | 2.0 ms | 3 | 0.30 mJ |
| **a3** | Local ACK + Selective Retransmit | 1 + selective NACKs | 3.0 ms | **4 retries** | 0.35 mJ |
| **a4** | QoS 2 Multi-step Handshake | 3 (PUBREC, PUBREL, PUBCOMP) | 5.0 ms | **5 retries** | 0.80 mJ |

---

## 4. DQN Controller & State Representation

- **Architecture**: `Input(8) -> Linear(64, ReLU) -> Linear(32, ReLU) -> Output(5)`
- **Optimizer**: Adam, learning rate $\alpha = 0.001$, discount $\gamma = 0.99$
- **Exploration**: $\epsilon$-greedy decaying from $1.0 \to 0.05$ over 50,000 steps

### 8 Normalized State Features:
1. $s_0$: Network utilization ($\rho \in [0, 1]$)
2. $s_1$: Rolling ACK-path packet loss rate ($\hat{p}_{\text{loss}} \in [0, 1]$)
3. $s_2$: Transmit queue occupancy ($\hat{Q} \in [0, 1]$)
4. $s_3$: Critical Control (CC) one-hot indicator ($\{0, 1\}$)
5. $s_4$: High-Bandwidth (HBS) one-hot indicator ($\{0, 1\}$)
6. $s_5$: Best-Effort (BET) one-hot indicator ($\{0, 1\}$)
7. $s_6$: Normalized remaining battery / source energy ($E_{\text{rem}} \in [0, 1]$)
8. $s_7$: Rolling average delivery latency ($\hat{L}_{\text{avg}} \in [0, 1]$)

---

## 5. Normalized Reward Function

$$\mathbf{r} = -0.40 \cdot \hat{L} - 0.20 \cdot \hat{O} - 0.30 \cdot V_{\text{rel}} + 0.10 \cdot (1 - \rho)$$

All weights sum strictly to $1.0$:
- **Normalized Latency ($\hat{L}$)**: $\min(L / D_{\text{tier}}, 2.0)$ where $D_{\text{tier}}$ is the tier deadline (10ms, 50ms, or 500ms).
- **Normalized Overhead ($\hat{O}$)**: $\min(\text{OverheadBytes} / \text{GLOBAL\_MAX\_OVERHEAD}, 1.0)$.
- **Reliability Violation ($V_{\text{rel}}$)**: Binary $1$ if delivery fails or deadline is missed; $0$ if delivered within deadline.
- **Utilization Bonus ($1 - \rho$)**: Incentivizes leaving breathing room for unexpected bursts.

---

## 6. Documented Deviations & Implementation Notes

1. **Simulator Execution**: Because native Windows lacks POSIX kernel primitives required by NS-3 without WSL, we developed a complete Python packet-level Discrete-Event Simulator (DES) implementing every single mathematical mechanism, packet queue, and Wi-Fi 6 channel contention model. C++ NS-3 v3.37 files are provided in `ns3/`.
2. **MQTT Abstraction**: Modeled as an application-level publish-subscribe message model with MQTT-equivalent semantics, distinguishing between native broker QoS and application ACK actions.
3. **Energy Measurement**: Modeled analytically based on radio transmission energy per byte and ACK handshake sequences rather than physical multimeter probes.
