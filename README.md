# RL-ACK: Adaptive Acknowledgment Framework for Smart Kitchen IoT Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation and evaluation of the **RL-ACK** framework based on the research paper:  
> *"RL-ACK: A Reinforcement Learning-Based Adaptive ACK Framework for Heterogeneous Smart Kitchen IoT Networks"*

---

## 🍳 Project Overview

Smart kitchen environments feature diverse IoT devices with conflicting network requirements:
- **Critical Control (CC)**: Gas leak detectors, emergency shutoff valves (Deadline: **10 ms**, requires ultra-high reliability).
- **High-Bandwidth Streaming (HBS)**: Pan vision cameras, thermal streaming (Deadline: **50 ms**).
- **Best-Effort Telemetry (BET)**: Routine ambient temperature, humidity sensors (Deadline: **500 ms**).

Standard static policies (such as fixed QoS 0 or fixed QoS 1/2) fail in congested or lossy wireless conditions—either flooding the channel with excessive control packets or failing to deliver critical alerts within their strict deadlines. 

**RL-ACK** solves this by deploying a **Deep Q-Network (DQN) agent** at the central **Edge Router**. The agent dynamically selects among five distinct acknowledgment actions ($\mathbf{a}_0$ through $\mathbf{a}_4$) based on live network utilization, ACK-path loss rates, queue occupancy, and message criticality.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               EDGE ROUTER                    │
                    │   ┌──────────────────────────────────────┐   │
                    │   │       RL-ACK DQN Controller          │   │
                    │   │  8-State Vector -> 64x32 -> 5 Actions│   │
                    │   └──────────────────────────────────────┘   │
                    │   ┌──────────────────────────────────────┐   │
                    │   │          MQTT Broker / AP            │   │
                    │   └──────────────────────────────────────┘   │
                    └──────────────────────┬───────────────────────┘
                                           │
                        IEEE 802.11ax Wi-Fi 6 (143.4 Mbps)
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                  │
 ┌──────┴──────┐                    ┌──────┴──────┐                    ┌──────┴──────┐
 │ Device 000  │                    │ Device 001  │                    │ Device 002  │
 │ (Critical)  │                    │ (Streaming) │                    │ (Telemetry) │
 │ Gas / Smoke │                    │ Camera / HBS│                    │ Temp / Hum  │
 └─────────────┘                    └─────────────┘                    └─────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone this repository and install dependencies:
```bash
cd RL_ACK_Smart_Kitchen
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (21/21 Unit & Integration Tests)
```bash
python tests/run_all_tests.py
```

### 3. Run the Gas Leak Emergency Demonstration
Simulates a kitchen gas spike, triggers the emergency alert, and demonstrates RL-ACK adaptation:
```bash
python experiments/gas_leak_demo.py
```

### 4. Launch the Interactive Web Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

### 5. Run Full Experiments Sweep
Execute experiments across device counts ($N \in \{10, 20, 30\}$), utilizations ($\rho \in [0.5, 0.9]$), loss rates, and seeds:
```bash
# Quick verification mode (96 runs across seeds):
python experiments/run_experiments.py --mode quick

# Aggregate results to CSVs:
python experiments/aggregate_results.py

# Evaluate all 6 policies under identical seeds:
python experiments/evaluate.py --devices 10 --util 0.7 --loss 0.05
```

---

## 📊 The Six Baseline Policies Compared

| Policy | Action Abstraction | Control Packets | Max Retries | Overhead | Latency | CC Reliability |
|---|---|---|---|---|---|---|
| **1. Static QoS 0** | Fire-and-forget ($\mathbf{a}_0$) | 0 | 0 | Lowest | Lowest | Vulnerable to loss |
| **2. Static Local ACK** | Edge proxy ACK ($\mathbf{a}_1$) | 1 | 0 | Low | Very Low | Moderate |
| **3. Static QoS 1** | PUBACK ($\mathbf{a}_2$) | 1 | 3 | High | Moderate | High |
| **4. Static Local+Selective**| Local ACK + NACK ($\mathbf{a}_3$) | 1+NACKs | 4 | Moderate | Low | Very High |
| **5. Static QoS 2** | 4-step Handshake ($\mathbf{a}_4$) | 3 | 5 | Highest | Highest | High overhead risk |
| **6. RL-ACK DQN** | **Adaptive Selection** | **Dynamic** | **Dynamic** | **-60% vs QoS 1** | **Sub-1ms** | **>99% Compliance** |

---

## 📁 Repository Organization

```
RL_ACK_Smart_Kitchen/
├── README.md                           # Master project documentation
├── requirements.txt                    # Python package dependencies
├── config/
│   └── default_config.json             # Simulation parameters, reward weights, sensor thresholds
├── ack/
│   └── actions.py                      # Implementation of the 5 ACK actions (a0 to a4)
├── sim/
│   ├── engine.py                       # Discrete-Event Simulation (DES) core clock & event queue
│   ├── channel.py                      # Wi-Fi 6 (802.11ax) propagation, loss & contention model
│   ├── node.py                         # Network nodes, roles, transmit queues
│   ├── packet.py                       # Packet metadata, headers, timestamps, delivery tracking
│   └── network.py                      # Orchestrator binding nodes, traffic, and RL controller
├── sensors/
│   ├── sensor_models.py                # Virtual sensors (gas, temp, humidity, smoke)
│   └── event_generator.py              # Poisson event generator for gas leaks & failures
├── traffic/
│   └── generator.py                    # Traffic generation across CC, HBS, and BET tiers
├── rl/
│   ├── dqn_agent.py                    # PyTorch Deep Q-Network (8 -> 64 -> 32 -> 5)
│   ├── replay_buffer.py                # Experience replay buffer with uniform sampling
│   ├── reward.py                       # Paper-aligned normalized reward function
│   └── controller_interface.py         # In-process and TCP socket IPC controller interface
├── policies/
│   └── baselines.py                    # 6 baseline policies for controlled comparisons
├── metrics/
│   └── collector.py                    # Comprehensive performance metrics collector
├── dashboard/
│   ├── app.py                          # Modern Streamlit web application
│   └── visualization.py                # Interactive Plotly chart generation
├── experiments/
│   ├── run_experiments.py              # Multi-parameter experiment runner with resume capability
│   ├── evaluate.py                     # Standalone policy evaluator under identical seeds
│   ├── aggregate_results.py            # Aggregates raw logs into CSV summaries
│   └── gas_leak_demo.py                # Reproducible gas-leak emergency demo script
├── tests/
│   ├── run_all_tests.py                # Master test runner
│   ├── test_traffic.py                 # Traffic classification and tier tests
│   ├── test_actions.py                 # Five ACK actions and retransmission tests
│   ├── test_dqn_and_reward.py          # DQN architecture and reward normalization tests
│   ├── test_sensors_and_alerts.py      # Sensor models and threshold alert tests
│   ├── test_simulation_and_metrics.py  # Simulation engine and metrics tests
│   └── test_reproducibility.py         # Determinism and random seed tests
├── ns3/
│   ├── README.md                       # WSL2 / Linux NS-3 v3.37 build guide
│   ├── scratch/rl-ack-kitchen.cc       # NS-3 v3.37 C++ simulation script
│   └── custom-modules/                 # NS-3 custom controller headers and C++ implementation
├── docs/
│   └── PAPER_ALIGNMENT.md              # Detailed mathematical and parameter paper mapping
└── results/
    ├── raw/                            # Raw per-run simulation JSON logs
    └── aggregated/                     # Aggregated summary CSV metrics
```
