// include/fluxpulse/mpi/ensemble_accumulator.hpp
#pragma once

#include <vector>

#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/mpi/mpi_types.hpp"

namespace fluxpulse {

struct Result;

namespace mpi {

class EnsembleAccumulator {
 public:
  EnsembleAccumulator() = default;

  EnsembleAccumulator(Index observable_count, Index sample_count);

  void Reset(Index observable_count, Index sample_count);

  void Clear() noexcept;

  bool empty() const noexcept { return flat_size_ == 0; }

  bool valid() const noexcept {
    return static_cast<Dim>(sum_.size()) == flat_size_ &&
           static_cast<Dim>(sum_sq_.size()) == flat_size_;
  }

  Index observable_count() const noexcept { return observable_count_; }

  Index sample_count() const noexcept { return sample_count_; }

  Dim flat_size() const noexcept { return flat_size_; }

  TrajectoryId trajectories() const noexcept { return trajectories_; }

  void set_trajectories(TrajectoryId n) noexcept { trajectories_ = n; }

  void AddTrajectoryCount(TrajectoryId n) noexcept { trajectories_ += n; }

  // Backward-compatible spelling used by the earlier header.
  void AddTrajectories(TrajectoryId n) noexcept { trajectories_ += n; }

  LocalEnsembleStatsView view() noexcept {
    LocalEnsembleStatsView v;
    v.trajectories = trajectories_;
    v.sum = Span<Real>(sum_.data(), flat_size_);
    v.sum_sq = Span<Real>(sum_sq_.data(), flat_size_);
    return v;
  }

  ConstLocalEnsembleStatsView view() const noexcept {
    ConstLocalEnsembleStatsView v;
    v.trajectories = trajectories_;
    v.sum = Span<const Real>(sum_.data(), flat_size_);
    v.sum_sq = Span<const Real>(sum_sq_.data(), flat_size_);
    return v;
  }

  Span<Real> sum() noexcept {
    return Span<Real>(sum_.data(), flat_size_);
  }

  Span<Real> sum_sq() noexcept {
    return Span<Real>(sum_sq_.data(), flat_size_);
  }

  Span<const Real> sum() const noexcept {
    return Span<const Real>(sum_.data(), flat_size_);
  }

  Span<const Real> sum_sq() const noexcept {
    return Span<const Real>(sum_sq_.data(), flat_size_);
  }

  void AccumulateTrajectory(Span<const Real> values) noexcept;

  void AccumulateTrajectory(const Real* values) noexcept;

  void Merge(ConstLocalEnsembleStatsView other) noexcept;

  void Merge(const EnsembleAccumulator& other) noexcept;

  bool FinalizeInto(Result* out) const;

 private:
  Index observable_count_ = 0;
  Index sample_count_ = 0;
  Dim flat_size_ = 0;

  TrajectoryId trajectories_ = 0;

  std::vector<Real> sum_;
  std::vector<Real> sum_sq_;
};

}  // namespace mpi
}  // namespace fluxpulse
