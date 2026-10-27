// include/fluxpulse/control/channel_id.hpp
#pragma once

#include <cstdint>

namespace fluxpulse {

// Strongly-typed, lightweight channel identifier.
// Use as a key for control terms and envelopes.
//
// Notes:
// - Trivially copyable and comparable.
// - Intentionally does not allocate and does not store strings.
// - If you want human-readable names, keep a separate mapping in I/O/config code.
struct ChannelId {
  uint16_t value = 0;

  constexpr ChannelId() = default;
  constexpr explicit ChannelId(uint16_t v) : value(v) {}

  static constexpr ChannelId Invalid() { return ChannelId{0}; }

  // Common channels for Phase 1 (suggested convention).
  static constexpr ChannelId DriveX() { return ChannelId{1}; }
  static constexpr ChannelId DriveY() { return ChannelId{2}; }
  static constexpr ChannelId DriveZ() { return ChannelId{3}; }

  constexpr bool valid() const { return value != 0; }

  friend constexpr bool operator==(ChannelId a, ChannelId b) { return a.value == b.value; }
  friend constexpr bool operator!=(ChannelId a, ChannelId b) { return a.value != b.value; }
  friend constexpr bool operator<(ChannelId a, ChannelId b) { return a.value < b.value; }
};

}  // namespace fluxpulse