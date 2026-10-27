// include/fluxpulse/mpi/mpi_types.hpp
#pragma once

#include <cstdint>

#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse::mpi {

// Scheduling policy for distributing trajectory work across MPI ranks.
enum class SchedulingMode : std::uint8_t {
  kStatic = 0,   // deterministic, fixed partitioning
  kDynamic = 1   // root assigns batches on demand
};

// Lightweight MPI run options.
// Keep this POD-like and free of any MPI-specific types.
struct MpiRunConfig {
  TrajectoryId total_trajectories = 0;
  Index batch_size = 1;

  SchedulingMode scheduling = SchedulingMode::kDynamic;
  bool reproducible = false;

  bool enable_trace = false;
  Index trace_rank = kInvalidIndex;

  Seed base_seed = 0;
};

// Dynamic-mode worker -> root request for more work.
struct BatchRequest {
  Index worker_rank = kInvalidIndex;
};

// Root -> worker assignment of a trajectory range.
struct BatchAssignment {
  TrajectoryId traj_begin = 0;
  Index traj_count = 0;

  // When true, worker should stop requesting/processing more work.
  bool terminate = false;
};

// Small metadata header describing one returned batch result.
struct BatchResultHeader {
  TrajectoryId traj_begin = 0;
  Index traj_count = 0;

  // Flattened result shape:
  // [observable_count * sample_count]
  Index observable_count = 0;
  Index sample_count = 0;
};

// Non-owning writable local statistics buffers.
// These views are useful for accumulation/reduction helpers without forcing
// ownership policy into this header.
struct LocalEnsembleStatsView {
  TrajectoryId trajectories = 0;
  Span<Real> sum;
  Span<Real> sum_sq;

  constexpr bool valid() const noexcept {
    return sum.size == sum_sq.size;
  }

  constexpr Dim flat_size() const noexcept {
    return sum.size;
  }
};

// Non-owning read-only local statistics buffers.
struct ConstLocalEnsembleStatsView {
  TrajectoryId trajectories = 0;
  Span<const Real> sum;
  Span<const Real> sum_sq;

  constexpr bool valid() const noexcept {
    return sum.size == sum_sq.size;
  }

  constexpr Dim flat_size() const noexcept {
    return sum.size;
  }
};

}  // namespace fluxpulse::mpi