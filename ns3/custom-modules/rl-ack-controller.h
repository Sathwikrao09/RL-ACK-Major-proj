/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef RL_ACK_CONTROLLER_H
#define RL_ACK_CONTROLLER_H

#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/nstime.h"
#include <map>
#include <string>

namespace ns3 {

enum RlAckAction {
    RL_ACK_A0 = 0, // QoS 0 without ACK
    RL_ACK_A1 = 1, // Local/Proxy ACK
    RL_ACK_A2 = 2, // QoS 1 PUBACK
    RL_ACK_A3 = 3, // Local ACK + Selective Retransmit
    RL_ACK_A4 = 4  // QoS 2 Multi-step Handshake
};

class RlAckController : public Object {
public:
    static TypeId GetTypeId(void);
    RlAckController();
    virtual ~RlAckController();

    // Select ACK action for incoming message
    RlAckAction DecideAction(uint32_t tier, double currentUtilization, double ackLossRate);

    // Apply action and compute retransmission limits
    uint32_t GetMaxRetries(RlAckAction action);
    uint32_t GetControlPacketsCount(RlAckAction action);
    double GetHandshakeDelayMs(RlAckAction action);

private:
    std::map<RlAckAction, uint32_t> m_actionCounts;
};

} // namespace ns3

#endif // RL_ACK_CONTROLLER_H
