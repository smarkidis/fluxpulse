// include/fluxpulse/core/types.hpp
#pragma once

#include <cstddef>   // std::size_t
#include <cstdint>   // fixed-width ints

// CUDA-readiness: allow annotating functions/types for host+device without
// pulling in CUDA headers. These macros are no-ops in normal C++ builds.
#if defined(__CUDACC__)
  #define FLUXPULSE_HD __host__ __device__
#else
  #define FLUXPULSE_HD
#endif

namespace fluxpulse {

// -----------------------------------------------------------------------------
// Scalar + time types
// -----------------------------------------------------------------------------
// IMPORTANT: choose ONE time unit project-wide (e.g., seconds or nanoseconds).
using Real = double;
using Time = double;

// -----------------------------------------------------------------------------
// Indices + sizes
// -----------------------------------------------------------------------------
// Use signed indices to avoid signed/unsigned warning cascades in loops.
// Convert to size_t only at the boundary to container access.
using Index = std::int64_t;
using Dim   = std::int64_t;
using Step  = std::int64_t;

// -----------------------------------------------------------------------------
// Randomness / trajectory identifiers
// -----------------------------------------------------------------------------
using Seed         = std::uint64_t;
using TrajectoryId = std::uint64_t;

// -----------------------------------------------------------------------------
// Common constants
// -----------------------------------------------------------------------------
constexpr Real kEpsilon = 1e-12;

constexpr Index kInvalidIndex = static_cast<Index>(-1);
constexpr Dim   kInvalidDim   = static_cast<Dim>(-1);

// -----------------------------------------------------------------------------
// Small helpers (header-only; no templates)
// -----------------------------------------------------------------------------
FLUXPULSE_HD constexpr std::size_t as_size_t(Index i) noexcept {
  return static_cast<std::size_t>(i);
}

FLUXPULSE_HD constexpr bool is_power_of_two(std::uint64_t x) noexcept {
  return x != 0u && (x & (x - 1u)) == 0u;
}

// 2^n as Dim (returns 0 if n is too large for uint64_t shifting).
FLUXPULSE_HD constexpr Dim pow2_u32(std::uint32_t n) noexcept {
  return (n >= 64u) ? Dim{0} : static_cast<Dim>(std::uint64_t{1} << n);
}

}  // namespace fluxpulse