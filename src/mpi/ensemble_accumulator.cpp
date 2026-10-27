// src/mpi/ensemble_accumulator.cpp
#include "fluxpulse/mpi/ensemble_accumulator.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>

#include "fluxpulse/sim/result.hpp"

namespace fluxpulse::mpi {

EnsembleAccumulator::EnsembleAccumulator(Index observable_count,
                                         Index sample_count) {
  Reset(observable_count, sample_count);
}

void EnsembleAccumulator::Reset(Index observable_count, Index sample_count) {
  if (observable_count < 0) observable_count = 0;
  if (sample_count < 0) sample_count = 0;

  observable_count_ = observable_count;
  sample_count_ = sample_count;
  flat_size_ = static_cast<Dim>(
      static_cast<std::size_t>(observable_count_) *
      static_cast<std::size_t>(sample_count_));

  trajectories_ = 0;

  sum_.assign(static_cast<std::size_t>(flat_size_), Real{0});
  sum_sq_.assign(static_cast<std::size_t>(flat_size_), Real{0});
}

void EnsembleAccumulator::Clear() noexcept {
  trajectories_ = 0;
  std::fill(sum_.begin(), sum_.end(), Real{0});
  std::fill(sum_sq_.begin(), sum_sq_.end(), Real{0});
}

void EnsembleAccumulator::AccumulateTrajectory(Span<const Real> values) noexcept {
  if (values.size != flat_size_) return;

  const std::size_t n = static_cast<std::size_t>(flat_size_);
  for (std::size_t i = 0; i < n; ++i) {
    const Real v = values.data[i];
    sum_[i] += v;
    sum_sq_[i] += v * v;
  }
  trajectories_ += 1;
}

void EnsembleAccumulator::AccumulateTrajectory(const Real* values) noexcept {
  if (!values && flat_size_ > 0) return;

  const std::size_t n = static_cast<std::size_t>(flat_size_);
  for (std::size_t i = 0; i < n; ++i) {
    const Real v = values[i];
    sum_[i] += v;
    sum_sq_[i] += v * v;
  }
  trajectories_ += 1;
}

void EnsembleAccumulator::Merge(ConstLocalEnsembleStatsView other) noexcept {
  if (!other.valid()) return;
  if (other.flat_size() != flat_size_) return;

  const std::size_t n = static_cast<std::size_t>(flat_size_);
  for (std::size_t i = 0; i < n; ++i) {
    sum_[i] += other.sum.data[i];
    sum_sq_[i] += other.sum_sq.data[i];
  }
  trajectories_ += other.trajectories;
}

void EnsembleAccumulator::Merge(const EnsembleAccumulator& other) noexcept {
  if (other.flat_size() != flat_size_) return;

  const std::size_t n = static_cast<std::size_t>(flat_size_);
  for (std::size_t i = 0; i < n; ++i) {
    sum_[i] += other.sum_[i];
    sum_sq_[i] += other.sum_sq_[i];
  }
  trajectories_ += other.trajectories_;
}

bool EnsembleAccumulator::FinalizeInto(Result* out) const {
  if (!out) return false;
  if (observable_count_ < 0 || sample_count_ < 0) return false;
  if (flat_size_ < 0) return false;
  if (sum_.size() != static_cast<std::size_t>(flat_size_)) return false;
  if (sum_sq_.size() != static_cast<std::size_t>(flat_size_)) return false;
  if (trajectories_ <= 0) return false;

  out->resize_with_uncertainty(observable_count_, sample_count_);

  const Real invN = Real{1} / static_cast<Real>(trajectories_);
  const std::size_t n = static_cast<std::size_t>(flat_size_);

  for (std::size_t i = 0; i < n; ++i) {
    const Real mean = sum_[i] * invN;
    Real var = sum_sq_[i] * invN - mean * mean;  // population variance
    if (var < Real{0}) var = Real{0};            // numerical clamp

    out->expectations[i] = mean;
    out->expectations_var[i] = var;
    out->expectations_stderr[i] = static_cast<Real>(
        std::sqrt(static_cast<double>(var * invN)));
  }

  return true;
}

}  // namespace fluxpulse::mpi