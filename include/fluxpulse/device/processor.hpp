// include/fluxpulse/device/processor.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/model_base.hpp"
#include "fluxpulse/sim/mcwf/mcwf_solver.hpp"
#include "fluxpulse/sim/result.hpp"

namespace fluxpulse {

// High-level simulation façade for Phase 1.
//
// Processor orchestrates:
// - pulse program (classical control coefficients u_k(t))
// - effective model (drift/control/collapse operators)
// - solver execution (MCWF trajectories)
// - trajectory averaging into a Result
//
// Phase 1 scope: single-process only (no MPI/thread pool yet).
struct RunOptions {
  // Record observables every `sample_stride` grid points (always includes last grid point).
  Dim sample_stride = 1;

  // Number of MCWF trajectories.
  uint64_t trajectories = 1;

  // Base seed for deterministic per-trajectory seeding.
  Seed base_seed = static_cast<Seed>(0);

  // Renormalize after each time step for stability.
  bool renormalize_each_step = true;

  // Optional MCWF diagnostics sampling (for trace counters, debugging).
  //
  // If diagnostics != nullptr and trajectories == 1, the solver fills:
  //   - norm2 (pre-renorm), p_total, jumped, jump_index
  // at stride `diagnostic_stride` (always includes last point).
  //
  // If trajectories > 1, Processor will ignore diagnostics (set to nullptr for solver).
  Dim diagnostic_stride = 1;               // >= 1
  mcwf::McwfDiagnostics* diagnostics = nullptr;  // optional
};

// Processor is intentionally a simple struct with explicit ownership.
struct Processor {
  ModelBase model;
  PulseProgram program;

  // Observables to measure (operators in full Hilbert space).
  std::vector<DenseOp> observables;
  std::vector<std::string> observable_names;

  RunOptions opt;

  // Validate basic consistency (dims, etc.).
  Status validate() const;

  // Run MCWF simulation and return averaged Result.
  //
  // Inputs:
  // - psi0: initial state vector, length = model.hilbert_dim
  //
  // Outputs:
  // - out: averaged expectations over trajectories (and, if trajectories>1, variance/stderr)
  Status run_mcwf(Span<const Complex> psi0, Result* out) const;
};

}  // namespace fluxpulse
