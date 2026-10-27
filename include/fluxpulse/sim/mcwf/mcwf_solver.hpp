// include/fluxpulse/sim/mcwf/mcwf_solver.hpp
#pragma once

#include <cstdint>

#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/model_base.hpp"
#include "fluxpulse/sim/result.hpp"
#include "fluxpulse/sim/mcwf/heff.hpp"
#include "fluxpulse/sim/mcwf/jump.hpp"
#include "fluxpulse/sim/mcwf/rk4.hpp"

namespace fluxpulse {
namespace mcwf {


// Compact log entry for every jump event (not downsampled).
struct McwfJumpEvent {
  Real t = Real{0};
  int32_t jump_index = -1;
};

// Optional diagnostics recorded during a trajectory run (useful for tracing/debugging).
//
// Sampling convention matches observables sampling:
// - recorded at grid indices 0, diagnostic_stride, 2*diagnostic_stride, ...
// - always includes last grid point
struct McwfDiagnostics {
  std::vector<Real> tlist;
  std::vector<Real> norm2;       // ||psi_propagated||^2 before renorm
  std::vector<Real> p_total;     // jump probability for step ending at this sample (last entry 0)
  std::vector<int32_t> jumped;   // 0/1 whether a jump occurred for that step
  std::vector<int32_t> jump_index; // collapse op index if jumped else -1

  // Full-resolution jump log: one entry per jump, regardless of diagnostic_stride.
  std::vector<McwfJumpEvent> jump_events;

  void clear() {
    tlist.clear();
    norm2.clear();
    p_total.clear();
    jumped.clear();
    jump_index.clear();
    jump_events.clear();
  }
};

// Options for one MCWF trajectory run.
struct McwfOptions {
  // Record observables every `sample_stride` grid points.
  // Phase 1 default: 1 (record at every grid point).
  Dim sample_stride = 1;

  // If true, renormalize after each non-Hermitian RK4 step (even if no jump).
  // Recommended for numerical stability.
  bool renormalize_each_step = true;


  // Diagnostics sampling:
  // - If diagnostics != nullptr, the solver will fill diagnostics arrays sampled every
  //   diagnostic_stride grid points (always includes last point).
  // - Recommended to use diagnostics only when running a single trajectory.
  Dim diagnostic_stride = 1;
  McwfDiagnostics* diagnostics = nullptr;
  
  std::vector<Complex>* final_state = nullptr;
};

// Workspace for running a single MCWF trajectory without per-step allocations.
//
// Lifetime / usage:
// - Call ws.init(model, program, n_obs) once.
// - Then reuse ws for repeated trajectory runs.
// - The solver will overwrite internal buffers each run.
struct McwfWorkspace {
  Dim dim = 0;
  Dim n_obs = 0;
  Dim n_collapse = 0;

  // State buffers
  std::vector<Complex> psi;
  std::vector<Complex> psi_next;

  // Observable buffers
  std::vector<Complex> obs_tmp;  // length dim
  std::vector<Real> obs_vals;    // length n_obs

  // RK4 buffers
  std::vector<Complex> k1;
  std::vector<Complex> k2;
  std::vector<Complex> k3;
  std::vector<Complex> k4;
  std::vector<Complex> rk_tmp;

  // Heff RHS buffers
  std::vector<Complex> heff_tmp;
  std::vector<Complex> heff_tmp_k;

  // Jump buffers
  std::vector<Complex> jump_tmp;   // C*psi scratch
  std::vector<Real> jump_probs;    // length n_collapse

  // PulseProgram cursor for monotonic envelope evaluation
  PulseProgram::Cursor program_cursor;

  // Precomputed K = sum_n C_n^† C_n (for Heff term). dim==0 if no collapse ops.
  DenseOp cdagc_sum;

  // Initialize buffers for a given model/program and number of observables.
  // This performs allocations, but it is NOT in the time-step hot path.
  Status init(const ModelBase& model, const PulseProgram& program, Dim n_observables);

  // Reset monotonic cursors (call at the start of each trajectory).
  void reset_cursors();
};

// Run a single MCWF trajectory and write results to `out`.
//
// Recorded times:
// - `out->tlist` will contain the TimeGrid points (possibly decimated by sample_stride).
// - expectations stored as out->expectations[obs * n_times + ti]
//
// Preconditions:
// - model.validate() is OK
// - program.grid.validate() is OK and program envelopes have been finalized
// - psi0.size == model.hilbert_dim and psi0 is normalized (approximately)
// - rng != nullptr
// - ws != nullptr and ws.init(...) called for matching dim/n_obs
Status run_mcwf_trajectory(const ModelBase& model,
                          const PulseProgram& program,
                          Span<const DenseOp> observables,
                          Span<const Complex> psi0,
                          Rng* rng,
                          const McwfOptions& opt,
                          McwfWorkspace* ws,
                          Result* out);

// A more memory-efficient variant for trajectory averaging:
// run one trajectory and add its expectations into `sum_expectations`.
// The caller later divides by number of trajectories.
//
// Layout:
// - sum_expectations must have size = observables.size * n_times
// - same time sampling as run_mcwf_trajectory (grid decimated by sample_stride)
//
// This function does NOT modify sum_expectations size; it only accumulates.
Status run_mcwf_trajectory_accumulate(const ModelBase& model,
                                     const PulseProgram& program,
                                     Span<const DenseOp> observables,
                                     Span<const Complex> psi0,
                                     Rng* rng,
                                     const McwfOptions& opt,
                                     McwfWorkspace* ws,
                                     Span<Real> sum_expectations);

}  // namespace mcwf
}  // namespace fluxpulse