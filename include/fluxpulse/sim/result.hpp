// include/fluxpulse/sim/result.hpp
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {

// Result is a simple container for time-series outputs produced by a simulation.
//
// Phase 1 scope:
// - store the time axis (tlist)
// - store expectation values for a set of observables at each time point
// - optionally store per-time/per-observable uncertainty estimates across trajectories
//
// Layout choice:
// - series are stored in a flat array in row-major form:
//     series[obs_index * n_times + time_index]
struct Result {
  // Time samples at which expectations are reported.
  std::vector<Real> tlist;

  // Optional names for each observable (same order as used when filling expectations).
  std::vector<std::string> observable_names;

  // Flat storage: size = n_obs * n_times (ensemble mean).
  std::vector<Real> expectations;

  // Optional uncertainty storage (same layout/size as expectations).
  // Variance across trajectories.
  std::vector<Real> expectations_var;

  // Standard error of the mean across trajectories: sqrt(var / Ntraj).
  std::vector<Real> expectations_stderr;

  // state vector result
  std::vector<Complex> final_state;

  Dim num_times() const { return static_cast<Dim>(tlist.size()); }
  Dim num_observables() const { return static_cast<Dim>(observable_names.size()); }

  bool has_final_state() const {
     return !final_state.empty();
  }

  void clear() {
    tlist.clear();
    observable_names.clear();
    expectations.clear();
    expectations_var.clear();
    expectations_stderr.clear();
    final_state.clear();
  }

  // Resizes storage for n_obs observables and n_times time points.
  // Does not fill values. Clears uncertainty fields.
  void resize(Dim n_obs, Dim n_times) {
    tlist.resize(static_cast<size_t>(n_times));
    observable_names.resize(static_cast<size_t>(n_obs));
    expectations.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_times));
    expectations_var.clear();
    expectations_stderr.clear();
    
    final_state.clear();
  }

  // Resizes storage and allocates uncertainty fields (var + stderr).
  void resize_with_uncertainty(Dim n_obs, Dim n_times) {
    resize(n_obs, n_times);
    expectations_var.resize(expectations.size());
    expectations_stderr.resize(expectations.size());
  }

  // Ensure uncertainty storage exists (allocates if needed).
  void ensure_uncertainty_storage() {
    if (expectations.empty()) {
      expectations_var.clear();
      expectations_stderr.clear();
      return;
    }
    if (expectations_var.size() != expectations.size()) {
      expectations_var.resize(expectations.size());
    }
    if (expectations_stderr.size() != expectations.size()) {
      expectations_stderr.resize(expectations.size());
    }
  }

  bool has_expect_var() const {
    return !expectations.empty() && expectations_var.size() == expectations.size();
  }
  bool has_expect_stderr() const {
    return !expectations.empty() && expectations_stderr.size() == expectations.size();
  }

  // Accessor for expectations (mutable / const).
  // Preconditions:
  // - obs_idx < num_observables()
  // - t_idx < num_times()
  Real& expect_at(Dim obs_idx, Dim t_idx) {
    const Dim nT = num_times();
    return expectations[static_cast<size_t>(obs_idx) * static_cast<size_t>(nT) +
                        static_cast<size_t>(t_idx)];
  }
  Real expect_at(Dim obs_idx, Dim t_idx) const {
    const Dim nT = num_times();
    return expectations[static_cast<size_t>(obs_idx) * static_cast<size_t>(nT) +
                        static_cast<size_t>(t_idx)];
  }

  // Accessors for variance / stderr. Callers should check has_expect_var()/has_expect_stderr().
  Real& expect_var_at(Dim obs_idx, Dim t_idx) {
    const Dim nT = num_times();
    return expectations_var[static_cast<size_t>(obs_idx) * static_cast<size_t>(nT) +
                            static_cast<size_t>(t_idx)];
  }
  Real expect_var_at(Dim obs_idx, Dim t_idx) const {
    const Dim nT = num_times();
    return expectations_var[static_cast<size_t>(obs_idx) * static_cast<size_t>(nT) +
                            static_cast<size_t>(t_idx)];
  }

  Real& expect_stderr_at(Dim obs_idx, Dim t_idx) {
    const Dim nT = num_times();
    return expectations_stderr[static_cast<size_t>(obs_idx) * static_cast<size_t>(nT) +
                               static_cast<size_t>(t_idx)];
  }
  Real expect_stderr_at(Dim obs_idx, Dim t_idx) const {
    const Dim nT = num_times();
    return expectations_stderr[static_cast<size_t>(obs_idx) * static_cast<size_t>(nT) +
                               static_cast<size_t>(t_idx)];
  }
};

}  // namespace fluxpulse
