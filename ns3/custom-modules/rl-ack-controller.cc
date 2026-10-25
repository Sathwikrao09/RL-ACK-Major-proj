/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "rl-ack-controller.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RlAckController");
NS_OBJECT_ENSURE_REGISTERED(RlAckController);

TypeId RlAckController::GetTypeId(void) {
    static TypeId tid = TypeId("ns3::RlAckController")
        .SetParent<Object>()
        .SetGroupName("Applications")
        .AddConstructor<RlAckController>();
    return tid;
}

RlAckController::RlAckController() {
    NS_LOG_FUNCTION(this);
}

RlAckController::~RlAckController() {
    NS_LOG_FUNCTION(this);
}

RlAckAction RlAckController::DecideAction(uint32_t tier, double currentUtilization, double ackLossRate) {
    // Tier 0 = Critical Control (CC)
    if (tier == 0) {
        if (ackLossRate > 0.08) {
            return RL_ACK_A3; // Local ACK with selective retransmission
        }
        return RL_ACK_A1; // Fast local ACK to guarantee sub-10ms latency
    }
    // Tier 1 = High-Bandwidth Streaming (HBS)
    else if (tier == 1) {
        if (currentUtilization > 0.8) {
            return RL_ACK_A1; // Offload channel
        }
        return RL_ACK_A2; // PUBACK
    }
    // Tier 2 = Best-Effort Telemetry (BET)
    else {
        if (currentUtilization > 0.7) {
            return RL_ACK_A0; // Fire-and-forget to preserve capacity
        }
        return RL_ACK_A1;
    }
}

uint32_t RlAckController::GetMaxRetries(RlAckAction action) {
    switch (action) {
        case RL_ACK_A0: return 0;
        case RL_ACK_A1: return 0;
        case RL_ACK_A2: return 3;
        case RL_ACK_A3: return 4;
        case RL_ACK_A4: return 5;
        default: return 0;
    }
}

uint32_t RlAckController::GetControlPacketsCount(RlAckAction action) {
    switch (action) {
        case RL_ACK_A0: return 0;
        case RL_ACK_A1: return 1;
        case RL_ACK_A2: return 1;
        case RL_ACK_A3: return 1;
        case RL_ACK_A4: return 3;
        default: return 0;
    }
}

double RlAckController::GetHandshakeDelayMs(RlAckAction action) {
    switch (action) {
        case RL_ACK_A0: return 0.0;
        case RL_ACK_A1: return 1.0;
        case RL_ACK_A2: return 2.0;
        case RL_ACK_A3: return 3.0;
        case RL_ACK_A4: return 5.0;
        default: return 0.0;
    }
}

} // namespace ns3
