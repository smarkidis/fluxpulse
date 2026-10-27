// src/mpi/trajectory_batcher.cpp
#include "fluxpulse/mpi/trajectory_batcher.hpp"

namespace fluxpulse::mpi {

BatchAssignment TrajectoryBatcher::StaticAssignment(TrajectoryId total_trajectories,
                                                    Index world_size,
                                                    Index rank) noexcept {
  BatchAssignment out{};

  if (total_trajectories <= 0) {
    out.terminate = true;
    return out;
  }
  if (world_size <= 0 || rank < 0 || rank >= world_size) {
    out.terminate = true;
    return out;
  }

  const TrajectoryId n = total_trajectories;
  const TrajectoryId p = static_cast<TrajectoryId>(world_size);
  const TrajectoryId r = static_cast<TrajectoryId>(rank);

  const TrajectoryId base = n / p;
  const TrajectoryId rem = n % p;

  const TrajectoryId count =
      base + ((r < rem) ? static_cast<TrajectoryId>(1) : static_cast<TrajectoryId>(0));

  const TrajectoryId begin =
      r * base + ((r < rem) ? r : rem);

  out.traj_begin = begin;
  out.traj_count = static_cast<Index>(count);
  out.terminate = (count <= 0);
  return out;
}

BatchAssignment TrajectoryBatcher::NextDynamicBatch(TrajectoryId total_trajectories,
                                                    Index batch_size,
                                                    TrajectoryId* next_traj) noexcept {
  BatchAssignment out{};

  if (!next_traj || total_trajectories <= 0) {
    out.terminate = true;
    return out;
  }

  const Index bs = NormalizeBatchSize(batch_size, 1);
  if (*next_traj >= total_trajectories) {
    out.terminate = true;
    return out;
  }

  const TrajectoryId begin = *next_traj;
  const TrajectoryId remaining = total_trajectories - begin;
  const TrajectoryId want = static_cast<TrajectoryId>(bs);
  const TrajectoryId count = (remaining < want) ? remaining : want;

  out.traj_begin = begin;
  out.traj_count = static_cast<Index>(count);
  out.terminate = (count <= 0);

  *next_traj = begin + count;
  return out;
}

Index TrajectoryBatcher::NumBatches(TrajectoryId total_trajectories,
                                    Index batch_size) noexcept {
  if (total_trajectories <= 0) return 0;

  const Index bs = NormalizeBatchSize(batch_size, 1);
  const TrajectoryId n = total_trajectories;
  const TrajectoryId b = static_cast<TrajectoryId>(bs);

  return static_cast<Index>((n + b - 1) / b);
}

Index TrajectoryBatcher::NormalizeBatchSize(Index requested_batch_size,
                                            Index fallback_batch_size) noexcept {
  if (requested_batch_size > 0) return requested_batch_size;
  if (fallback_batch_size > 0) return fallback_batch_size;
  return 1;
}

SchedulingMode TrajectoryBatcher::EffectiveScheduling(const MpiRunConfig& cfg) noexcept {
  if (cfg.reproducible) return SchedulingMode::kStatic;
  return cfg.scheduling;
}

}  // namespace fluxpulse::mpi