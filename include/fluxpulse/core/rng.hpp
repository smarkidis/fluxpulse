// include/fluxpulse/core/rng.hpp
#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// ============================================================================
// RNG backend interface (C-like, no templates, no exceptions)
// ============================================================================

struct RngBackend {
  void (*seed)(void* ctx, Seed s) noexcept;
  std::uint64_t (*next_u64)(void* ctx) noexcept;
};

// ============================================================================
// Default backend: SplitMix64
// ============================================================================

struct SplitMix64State {
  std::uint64_t state = 0u;
};

inline void splitmix64_seed(void* ctx, Seed s) noexcept {
  SplitMix64State* st = static_cast<SplitMix64State*>(ctx);
  st->state = static_cast<std::uint64_t>(s);
}

inline std::uint64_t splitmix64_next_u64(void* ctx) noexcept {
  SplitMix64State* st = static_cast<SplitMix64State*>(ctx);
  std::uint64_t z = (st->state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline const RngBackend* splitmix64_backend() noexcept {
  static const RngBackend kBackend = {&splitmix64_seed, &splitmix64_next_u64};
  return &kBackend;
}

// ============================================================================
// RNG front-end used everywhere in FluxPulse
// ============================================================================

struct Rng {
  SplitMix64State splitmix64_state{};
  const RngBackend* backend = nullptr;
  void* ctx = nullptr;

  void init_splitmix64() noexcept {
    backend = splitmix64_backend();
    ctx = static_cast<void*>(&splitmix64_state);
  }

  // Caller owns external_state and must keep it alive.
  void init_external(const RngBackend* external_backend,
                     void* external_state) noexcept {
    backend = external_backend;
    ctx = external_state;
  }

  bool is_initialized() const noexcept { return backend != nullptr && ctx != nullptr; }

  void seed(Seed s) noexcept { backend->seed(ctx, s); }
  std::uint64_t next_u64() noexcept { return backend->next_u64(ctx); }
  std::uint32_t next_u32() noexcept { return static_cast<std::uint32_t>(next_u64() >> 32); }

  Real uniform01() noexcept {
    constexpr std::uint64_t kMask53 = (std::uint64_t{1} << 53) - 1;
    const std::uint64_t r = next_u64() & kMask53;
    return static_cast<Real>(r) / static_cast<Real>(std::uint64_t{1} << 53);
  }

  Real uniform01_open() noexcept {
    Real u = uniform01();
    while (u <= 0.0) u = uniform01();
    return u;
  }

  Real uniform_real(Real a, Real b) noexcept { return a + (b - a) * uniform01(); }

  std::uint64_t uniform_u64(std::uint64_t lo, std::uint64_t hi) noexcept {
    if (lo > hi) { const std::uint64_t tmp = lo; lo = hi; hi = tmp; }
    const std::uint64_t range = hi - lo + 1u;
    if (range == 0u) return next_u64();

    const std::uint64_t limit =
        std::numeric_limits<std::uint64_t>::max() -
        (std::numeric_limits<std::uint64_t>::max() % range);

    std::uint64_t x = next_u64();
    while (x > limit) x = next_u64();
    return lo + (x % range);
  }

  std::int64_t uniform_i64(std::int64_t lo, std::int64_t hi) noexcept {
    if (lo > hi) { const std::int64_t tmp = lo; lo = hi; hi = tmp; }
    return static_cast<std::int64_t>(
        uniform_u64(static_cast<std::uint64_t>(lo), static_cast<std::uint64_t>(hi)));
  }

  bool bernoulli(Real p) noexcept {
    if (p <= 0.0) return false;
    if (p >= 1.0) return true;
    return uniform01() < p;
  }

  Real exponential(Real rate) noexcept {
    const Real u = uniform01_open();
    return -std::log(u) / rate;
  }
};

// Deterministic seed derivation for trajectory i (MPI-size invariant).
inline Seed derive_trajectory_seed(Seed base_seed, TrajectoryId traj_id) noexcept {
  std::uint64_t x = static_cast<std::uint64_t>(base_seed);
  x ^= static_cast<std::uint64_t>(traj_id) + 0x9E3779B97F4A7C15ULL + (x << 6) + (x >> 2);
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31);
  return static_cast<Seed>(x);
}

}  // namespace fluxpulse