// include/fluxpulse/sim/lindblad/lindblad_solver.hpp
#pragma once

#include <cstdint>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/model/model_base.hpp"
#include "fluxpulse/sim/result.hpp"

namespace fluxpulse {
namespace lindblad {

// Options for Lindblad master-equation evolution.
//
// Phase 1 design goal:
// - small, deterministic reference solver for validating MCWF on 1–2 qubits (optionally 3).
// - correctness > performance. O(D^3) dense math is fine for D <= 32.
struct LindbladOptions {
  // Record observables every k-th grid point, and always include the last point.
  Dim sample_stride = 1;

  // If true, renormalize rho by its trace after each step:
  //   rho <- rho / Tr(rho)
  // This helps prevent trace drift from numerical integration error.
  bool enforce_trace_one = true;

  // If true, symmetrize rho after each step:
  //   rho <- (rho + rho^†)/2
  // Useful to mitigate numerical Hermiticity drift (optional).
  bool enforce_hermiticity = false;
};

// Workspace for Lindblad solver.
// - Holds scratch buffers to avoid reallocations.
// - Precomputes C† and C†C for collapse ops, and drift Hamiltonian sum.
struct LindbladWorkspace {
  Dim dim = 0;

  // Precomputed: H_drift = sum_j drift_terms[j]
  DenseOp H_drift;

  // Precomputed per collapse operator:
  // - Cdag[j] = C_j^†
  // - CdC[j]  = C_j^† C_j
  std::vector<DenseOp> Cdag;
  std::vector<DenseOp> CdC;

  // Scratch matrices (stored as flat column-major arrays of size dim*dim).
  // We keep them as raw vectors to avoid DenseOp overhead when doing many temporary multiplies.
  std::vector<Complex> rho;    // current rho
  std::vector<Complex> k1;
  std::vector<Complex> k2;
  std::vector<Complex> k3;
  std::vector<Complex> k4;
  // RK4 stage input must not alias the RHS multiplication scratch buffers.
  std::vector<Complex> rho_stage;
  std::vector<Complex> tmp1;
  std::vector<Complex> tmp2;
  std::vector<Complex> tmp3;

  // Build / resize workspace for a specific model.
  Status init(const ModelBase& model);

  void clear() {
    dim = 0;
    H_drift = DenseOp();
    Cdag.clear();
    CdC.clear();
    rho.clear();
    k1.clear();
    k2.clear();
    k3.clear();
    k4.clear();
    rho_stage.clear();
    tmp1.clear();
    tmp2.clear();
    tmp3.clear();
  }
};

// Run Lindblad master equation evolution with RK4 on the TimeGrid of `program`.
//
// Inputs:
// - model: ModelBase containing drift/control/collapse operators (full-system DenseOp).
// - program: PulseProgram (provides u_k(t) via envelopes).
// - observables: list of full-system operators O_k (DenseOp dim==model.hilbert_dim).
// - rho0: initial density matrix in column-major order, size dim*dim.
//         (rho0 should be Hermitian PSD with trace 1; we do not enforce PSD.)
// - opt: solver options.
// - ws: workspace (must be initialized for this model via ws->init(model)).
//
// Output:
// - out: Result with expectations Tr(O rho(t)) sampled on the grid.
//        We take the real part of the trace (imag parts should be ~0 for Hermitian O).
Status run_lindblad(const ModelBase& model,
                    const PulseProgram& program,
                    Span<const DenseOp> observables,
                    Span<const Complex> rho0,
                    const LindbladOptions& opt,
                    LindbladWorkspace* ws,
                    Result* out);

// Helper: build a pure-state density matrix rho = |psi><psi|.
// - psi: length dim
// - out_rho: resized to dim*dim, column-major.
Status rho_from_psi(Span<const Complex> psi, std::vector<Complex>* out_rho);

}  // namespace lindblad
}  // namespace fluxpulse
