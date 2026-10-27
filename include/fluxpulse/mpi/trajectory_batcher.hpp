// include/fluxpulse/mpi/trajectory_batcher.hpp
#pragma once

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/mpi/mpi_types.hpp"

namespace fluxpulse::mpi {

// Stateless helper for partitioning trajectory work across MPI ranks.
//
// Design goals:
// - keep batching policy separate from MCWF solver internals
// - support both reproducible static partitioning and dynamic batching
// - avoid <mpi.h> in the header
//
// Conventions:
// - trajectory ranges are half-open: [traj_begin, traj_begin + traj_count)
// - root rank is assumed to be rank 0
// - in static mode, rank 0 may also compute a local batch if desired
class TrajectoryBatcher {
 public:
  // Compute a static batch assignment for one rank.
  //
  // This partitions [0, total_trajectories) across `world_size` ranks as evenly
  // as possible. Lower ranks receive one extra trajectory when the division is
  // not exact.
  //
  // Preconditions:
  // - world_size > 0
  // - 0 <= rank < world_size
  static BatchAssignment StaticAssignment(TrajectoryId total_trajectories,
                                          Index world_size,
                                          Index rank) noexcept;

  // Compute the next contiguous batch in a serial/dynamic assignment stream.
  //
  // Typical root-side usage:
  //   TrajectoryId next = 0;
  //   while (...) {
  //     BatchAssignment a =
  //         TrajectoryBatcher::NextDynamicBatch(total, batch_size, &next);
  //   }
  //
  // Behavior:
  // - returns terminate=true when no trajectories remain
  // - otherwise returns a non-empty half-open range
  //
  // Preconditions:
  // - next_traj != nullptr
  // - batch_size > 0
  static BatchAssignment NextDynamicBatch(TrajectoryId total_trajectories,
                                          Index batch_size,
                                          TrajectoryId* next_traj) noexcept;

  // Number of batches needed to cover total_trajectories with a fixed batch size.
  //
  // Returns 0 when total_trajectories <= 0.
  //
  // Preconditions:
  // - batch_size > 0
  static Index NumBatches(TrajectoryId total_trajectories,
                          Index batch_size) noexcept;

  // Clamp/normalize batch size for safe use.
  //
  // Current policy:
  // - if requested_batch_size <= 0, return fallback_batch_size
  // - if fallback_batch_size <= 0, use 1
  static Index NormalizeBatchSize(Index requested_batch_size,
                                  Index fallback_batch_size = 1) noexcept;

  // True if the assignment encodes no work.
  static bool IsEmpty(const BatchAssignment& a) noexcept {
    return a.terminate || a.traj_count <= 0;
  }

  // End trajectory id of a half-open assignment range.
  static TrajectoryId End(const BatchAssignment& a) noexcept {
    return a.traj_begin + static_cast<TrajectoryId>(a.traj_count);
  }

  // Suggested scheduling mode from run config.
  //
  // Current convention:
  // - reproducible => static
  // - otherwise use cfg.scheduling
  static SchedulingMode EffectiveScheduling(const MpiRunConfig& cfg) noexcept;
};

}  // namespace fluxpulse::mpi