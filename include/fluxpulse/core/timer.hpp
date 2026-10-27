// include/fluxpulse/core/timer.hpp
#pragma once

#include <chrono>
#include <cstdint>

#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// Simple wall-clock timer (monotonic).
// Header-only “leaf” utility: safe to include widely.
struct Timer {
  using Clock = std::chrono::steady_clock;

  Clock::time_point t0{};

  // Starts (or restarts) the timer.
  void reset() noexcept { t0 = Clock::now(); }

  // Seconds since last reset.
  Real elapsed_seconds() const noexcept {
    const auto dt = Clock::now() - t0;
    return std::chrono::duration<Real>(dt).count();
  }

  // Nanoseconds since last reset.
  std::uint64_t elapsed_ns() const noexcept {
    const auto dt = Clock::now() - t0;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
  }
};

// Convenience: current steady-clock time as nanoseconds since an arbitrary epoch.
// Useful for lightweight timestamps in logs.
inline std::uint64_t now_ns() noexcept {
  const auto tp = Timer::Clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(tp).count());
}

}  // namespace fluxpulse