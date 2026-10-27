// src/device/processor.cpp
#include "fluxpulse/device/processor.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace fluxpulse {
namespace {

Status validate_observables(const ModelBase& model,
                            const std::vector<DenseOp>& observables,
                            const std::vector<std::string>& names) {
  const Dim d = model.hilbert_dim;
  for (size_t k = 0; k < observables.size(); ++k) {
    if (observables[k].dim() != d) {
      return Error(ErrorCode::kInvalidArgument, "Processor: observable dim mismatch");
    }
  }
  if (!names.empty() && names.size() != observables.size()) {
    return Error(ErrorCode::kInvalidArgument,
                 "Processor: observable_names must match observables size (or be empty)");
  }
  return Ok();
}

// Match mcwf_solver sampling convention:
// record at grid indices 0, stride, 2*stride, ... and always include last grid point.
inline Dim compute_num_samples(Dim n_points, Dim stride) {
  if (n_points <= 0) return 0;
  if (stride <= 0) stride = 1;
  const Dim last = n_points - 1;
  Dim n = last / stride + 1;
  if ((last % stride) != 0) n += 1;
  return n;
}

}  // namespace

Status Processor::validate() const {
  FLUXPULSE_RETURN_IF_ERROR(model.validate());

  // PulseProgram::validate is non-const; validate a copy (finalize_envelopes=false).
  PulseProgram tmp = program;
  FLUXPULSE_RETURN_IF_ERROR(tmp.validate(/*finalize_envelopes=*/false));

  FLUXPULSE_RETURN_IF_ERROR(validate_observables(model, observables, observable_names));

  if (opt.sample_stride <= 0) return Error(ErrorCode::kInvalidArgument, "Processor: sample_stride must be >= 1");
  if (opt.trajectories == 0) return Error(ErrorCode::kInvalidArgument, "Processor: trajectories must be >= 1");
  if (opt.diagnostic_stride <= 0) {
    return Error(ErrorCode::kInvalidArgument, "Processor: diagnostic_stride must be >= 1");
  }

  return Ok();
}

Status Processor::run_mcwf(Span<const Complex> psi0, Result* out) const {
  if (!out) return Error(ErrorCode::kInvalidArgument, "Processor::run_mcwf: out is null");

  // A final pure state is only meaningful for a single MCWF trajectory.
  // Clear any state left over from a previous run before starting.
  out->final_state.clear();

  FLUXPULSE_RETURN_IF_ERROR(validate());

  const Dim d = model.hilbert_dim;
  if (psi0.size != d) {
    return Error(ErrorCode::kInvalidArgument, "Processor::run_mcwf: psi0 dimension mismatch");
  }

  const Dim n_obs = static_cast<Dim>(observables.size());
  const Dim n_points = program.grid.size();

  mcwf::McwfWorkspace ws;
  FLUXPULSE_RETURN_IF_ERROR(ws.init(model, program, n_obs));

  mcwf::McwfOptions mopt;
  mopt.sample_stride = opt.sample_stride;
  mopt.renormalize_each_step = opt.renormalize_each_step;

  // Diagnostics: only meaningful for trajectories==1 for trace/debug usage.
  mopt.diagnostic_stride = opt.diagnostic_stride;
  mopt.diagnostics = (opt.trajectories == 1) ? opt.diagnostics : nullptr;

  // A state vector represents one pure trajectory.  Capture it only when
  // exactly one trajectory is requested.  Multi-trajectory MCWF represents
  // an ensemble and therefore has no unique final state vector.
  mopt.final_state = (opt.trajectories == 1) ? &out->final_state : nullptr;

  // Single trajectory: run directly.
  if (opt.trajectories == 1) {
    const Seed seed = derive_trajectory_seed(opt.base_seed, static_cast<TrajectoryId>(0));
    Rng rng;
    rng.init_splitmix64();
    rng.seed(seed);

    FLUXPULSE_RETURN_IF_ERROR(
        mcwf::run_mcwf_trajectory(
            model, program,
            Span<const DenseOp>(observables.data(), n_obs),
            psi0,
            &rng, mopt, &ws,
            out));

    // Attach observable names
    if (!observable_names.empty()) {
      out->observable_names = observable_names;
    }
    return Ok();
  }

  // Multi-trajectory mean + variance/stderr.
  const Dim stride = (opt.sample_stride <= 0) ? 1 : opt.sample_stride;
  const Dim n_samples = compute_num_samples(n_points, stride);

  std::vector<Real> sum(static_cast<size_t>(n_obs) * static_cast<size_t>(n_samples), Real{0});
  std::vector<Real> sum_sq(static_cast<size_t>(n_obs) * static_cast<size_t>(n_samples), Real{0});

  // Ensure single-trajectory-only outputs are disabled in ensemble mode.
  mcwf::McwfOptions mopt_no_diag = mopt;
  mopt_no_diag.diagnostics = nullptr;
  mopt_no_diag.final_state = nullptr;

  // Reuse a per-trajectory Result container to avoid repeated allocations.
  Result traj;
  traj.resize(n_obs, n_samples);

  for (uint64_t tr = 0; tr < opt.trajectories; ++tr) {
    const Seed seed = derive_trajectory_seed(opt.base_seed, static_cast<TrajectoryId>(tr));
    Rng rng;
    rng.init_splitmix64();
    rng.seed(seed);

    FLUXPULSE_RETURN_IF_ERROR(
        mcwf::run_mcwf_trajectory(
            model, program,
            Span<const DenseOp>(observables.data(), n_obs),
            psi0,
            &rng, mopt_no_diag, &ws,
            &traj));

    // Accumulate moments
    for (size_t i = 0; i < traj.expectations.size(); ++i) {
      const Real x = traj.expectations[i];
      sum[i] += x;
      sum_sq[i] += x * x;
    }
  }

  out->resize_with_uncertainty(n_obs, n_samples);

  // Fill tlist consistent with solver sampling convention.
  for (Dim i = 0; i < n_samples; ++i) {
    Dim grid_i = i * stride;
    if (grid_i >= n_points) grid_i = n_points - 1;
    out->tlist[static_cast<size_t>(i)] = program.grid.t(grid_i);
  }

  // Attach observable names
  if (!observable_names.empty()) {
    out->observable_names = observable_names;
  }

  const Real invN = Real{1} / static_cast<Real>(opt.trajectories);
  for (size_t i = 0; i < sum.size(); ++i) {
    const Real mean = sum[i] * invN;
    Real var = sum_sq[i] * invN - mean * mean;  // population variance
    if (var < Real{0}) var = Real{0};  // numerical clamp
    out->expectations[i] = mean;
    out->expectations_var[i] = var;
    out->expectations_stderr[i] = std::sqrt(var * invN);
  }

  return Ok();
}

}  // namespace fluxpulse
