# NS-3 (v3.37) RL-ACK Simulation Scaffolding

This directory contains the C++ simulation scripts and custom modules for running the **RL-ACK Smart Kitchen IoT Simulator** in native **NS-3 (v3.37)** under Linux or WSL2.

---

## Prerequisites (WSL2 / Ubuntu 22.04 LTS)

Because NS-3 relies on POSIX system APIs (`fork()`, POSIX threads, Unix domain sockets), running it on Windows requires **WSL2**:

```bash
# 1. Install prerequisites in Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install -y g++ python3 cmake ninja-build git python3-dev pkg-config sqlite3 libsqlite3-dev

# 2. Download and unpack ns-3.37
wget https://www.nsnam.org/releases/ns-allinone-3.37.tar.bz2
tar -xjf ns-allinone-3.37.tar.bz2
cd ns-allinone-3.37/ns-3.37
```

---

## Building and Running the Simulation

```bash
# 1. Copy the kitchen simulation script into NS-3 scratch directory
cp /path/to/RL_ACK_Smart_Kitchen/ns3/scratch/rl-ack-kitchen.cc scratch/

# 2. Copy the custom controller module into src/applications/model/
cp /path/to/RL_ACK_Smart_Kitchen/ns3/custom-modules/rl-ack-controller.h src/applications/model/
cp /path/to/RL_ACK_Smart_Kitchen/ns3/custom-modules/rl-ack-controller.cc src/applications/model/

# 3. Configure and build NS-3 with CMake/Ninja
./ns3 configure --build-profile=optimized --enable-examples
./ns3 build

# 4. Run the simulation
./ns3 run "scratch/rl-ack-kitchen --nDevices=10 --duration=300 --ackLoss=0.05 --utilization=0.7"
```

---

## Python DQN Controller Integration (TCP Socket IPC)

To run NS-3 with the live PyTorch Deep Q-Network:

1. In Windows or Linux, start the Python controller server:
   ```bash
   python -c "
   from rl.controller_interface import SocketControllerServer, LocalController
   from rl.dqn_agent import DQNAgent
   from policies.baselines import DQNPolicy
   import json

   with open('config/default_config.json') as f:
       cfg = json.load(f)
   agent = DQNAgent(cfg['dqn'])
   agent.load_checkpoint('models/dqn_checkpoint_latest.pt')
   controller = LocalController(DQNPolicy(agent))
   server = SocketControllerServer(controller, host='0.0.0.0', port=5555)
   server.start()
   "
   ```

2. Run NS-3 with `--useDqnIpc=true`:
   ```bash
   ./ns3 run "scratch/rl-ack-kitchen --nDevices=20 --useDqnIpc=true"
   ```
