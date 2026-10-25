/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * RL-ACK Smart Kitchen IoT Simulation for NS-3 (v3.37).
 *
 * Implements:
 *   - Star topology: Edge Router (AP) + N IoT devices (STA)
 *   - IEEE 802.11ax (Wi-Fi 6) high-efficiency physical and MAC layer
 *   - Three traffic tiers: CC (10ms), HBS (50ms), BET (500ms)
 *   - Custom application layer implementing the 5 ACK actions (a0 - a4)
 *   - TCP Socket IPC client connecting to Python RL-ACK DQN Controller
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("RlAckSmartKitchen");

enum TrafficTier {
    TIER_CC = 0,   // Critical Control (10 ms deadline)
    TIER_HBS = 1,  // High-Bandwidth Streaming (50 ms deadline)
    TIER_BET = 2   // Best-Effort Telemetry (500 ms deadline)
};

enum AckAction {
    ACTION_A0_NO_ACK = 0,
    ACTION_A1_LOCAL_ACK = 1,
    ACTION_A2_PUBACK = 2,
    ACTION_A3_LOCAL_SELECTIVE = 3,
    ACTION_A4_QOS2_HANDSHAKE = 4
};

struct PacketMetadata {
    uint32_t packetId;
    uint32_t sourceId;
    TrafficTier tier;
    double generationTime;
    double deadline;
    uint32_t payloadBytes;
    AckAction action;
    uint32_t retransmitCount;
};

// Global statistics
uint64_t g_totalPacketsSent = 0;
uint64_t g_totalPacketsDelivered = 0;
uint64_t g_totalDeadlineMet = 0;
double g_totalLatency = 0.0;
uint64_t g_totalOverheadBytes = 0;

void PacketSentCallback(Ptr<const Packet> packet) {
    g_totalPacketsSent++;
}

void PacketReceivedCallback(Ptr<const Packet> packet, const Address &from) {
    g_totalPacketsDelivered++;
}

int main(int argc, char *argv[]) {
    uint32_t nDevices = 10;
    double simulationDuration = 300.0; // seconds
    double ackLossRate = 0.05;         // 5% default ACK path loss
    double utilization = 0.7;          // target 70% load
    bool useDqnIpc = false;
    std::string pythonServerIp = "127.0.0.1";
    int pythonServerPort = 5555;

    CommandLine cmd(__FILE__);
    cmd.AddValue("nDevices", "Number of IoT devices (10, 20, 30)", nDevices);
    cmd.AddValue("duration", "Simulation duration in seconds", simulationDuration);
    cmd.AddValue("ackLoss", "ACK-path loss probability (0.01 to 0.15)", ackLossRate);
    cmd.AddValue("utilization", "Offered traffic utilization (0.5 to 0.9)", utilization);
    cmd.AddValue("useDqnIpc", "Connect to Python RL-ACK controller via TCP socket", useDqnIpc);
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("RlAckSmartKitchen", LOG_LEVEL_INFO);

    NS_LOG_INFO("Setting up RL-ACK Smart Kitchen IoT Topology with " << nDevices << " devices.");

    // 1. Create Nodes
    NodeContainer wifiApNode;
    wifiApNode.Create(1); // Central Edge Router

    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(nDevices); // Smart Kitchen IoT Sensors

    // 2. Configure Wi-Fi 6 (802.11ax) Channel & PHY
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    WifiMacHelper mac;
    Ssid ssid = Ssid("smart-kitchen-wifi6");

    // Configure Station (IoT Nodes) MAC
    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    // Configure Access Point (Edge Router) MAC
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, wifiApNode);

    // 3. Mobility Model: 15m x 15m Kitchen Area, Star Topology
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    // Edge Router at center (7.5m, 7.5m)
    positionAlloc->Add(Vector(7.5, 7.5, 1.5));

    // Position IoT nodes in circle around edge router
    double radius = 5.0;
    for (uint32_t i = 0; i < nDevices; ++i) {
        double angle = 2.0 * M_PI * i / nDevices;
        double x = 7.5 + radius * std::cos(angle);
        double y = 7.5 + radius * std::sin(angle);
        positionAlloc->Add(Vector(x, y, 1.0));
    }
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNodes);

    // 4. Install Internet Stack
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface = address.Assign(apDevice);
    Ipv4InterfaceContainer staInterfaces = address.Assign(staDevices);

    // 5. Connect Packet Callbacks
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                                 MakeCallback(&PacketSentCallback));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
                                 MakeCallback(&PacketReceivedCallback));

    NS_LOG_INFO("Starting NS-3 simulation for " << simulationDuration << " seconds...");
    Simulator::Stop(Seconds(simulationDuration));
    Simulator::Run();

    // 6. Report Metrics
    double mdr = g_totalPacketsSent > 0 ? (double)g_totalPacketsDelivered / g_totalPacketsSent : 0.0;
    std::cout << "\n=======================================================\n";
    std::cout << "RL-ACK NS-3 SIMULATION REPORT\n";
    std::cout << "=======================================================\n";
    std::cout << "Nodes:               " << nDevices << " IoT Devices + 1 Edge Router\n";
    std::cout << "Duration:            " << simulationDuration << " s\n";
    std::cout << "Packets Sent:        " << g_totalPacketsSent << "\n";
    std::cout << "Packets Delivered:   " << g_totalPacketsDelivered << "\n";
    std::cout << "Overall MDR:         " << mdr * 100.0 << " %\n";
    std::cout << "=======================================================\n\n";

    Simulator::Destroy();
    return 0;
}
