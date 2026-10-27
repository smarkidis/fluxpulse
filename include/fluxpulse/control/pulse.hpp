// include/fluxpulse/control/pulse.hpp
#pragma once

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/control/envelope.hpp"

namespace fluxpulse {

// A Pulse binds an Envelope to a ChannelId.
// Conceptually this defines the coefficient u_k(t) for that channel.
//
// Phase 1 convention:
// - A PulseProgram typically has at most one Pulse per channel (enforced/checked
//   at the PulseProgram layer).
struct Pulse {
  ChannelId channel;
  Envelope envelope;

  Pulse() = default;

  Pulse(ChannelId ch, Envelope env) : channel(ch), envelope(std::move(env)) {}
};

}  // namespace fluxpulse