// include/fluxpulse/sim/mcwf/heff.hpp
#pragma once

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/model_base.hpp"

namespace fluxpulse {
namespace mcwf {

// MCWF effective (non-Hermitian) evolution:
//
//   d|psi>/dt = -i H(t)|psi> - 0.5 * (Σ_n C_n† C_n) |psi>
//
// where
//   H(t) = Σ_j H_drift[j] + Σ_k u_k(t) H_control[k].
//
// This header defines:
// - a helper to precompute K = Σ C†C
// - an RK4-compatible RHS callback that computes dpsi for Heff evolution
//
// Performance notes (Phase 1):
// - caller provides scratch buffers (no allocations in hot path)
// - PulseProgram cursor is used for amortized O(1) envelope evaluation in time loops

// Workspace for Heff RHS evaluation.
// All spans must have length == model.hilbert_dim.
struct HeffWorkspace {
  // Scratch for applying Hamiltonian terms (DenseOp * psi).
  Span<Complex> tmp;

  // Scratch for applying K = Σ C†C (DenseOp * psi) if enabled.
  Span<Complex> tmp_k;
};

// Precompute K = Σ_n (C_n† C_n).
// If there are no collapse operators, out_K is set to dim=0 and Ok() is returned.
Status build_cdagc_sum(const ModelBase& model, DenseOp* out_K);

// User/context passed to the Heff RHS callback.
struct HeffRhsUser {
  const ModelBase* model = nullptr;
  const PulseProgram* program = nullptr;

  // Optional: enable monotonic envelope evaluation.
  // If nullptr, coeff() (binary-search) evaluation is used.
  PulseProgram::Cursor* program_cursor = nullptr;

  // Optional: precomputed K = Σ C†C (same dim as model.hilbert_dim).
  // If nullptr or dim==0, the -0.5*K|psi> term is omitted.
  const DenseOp* cdagc_sum = nullptr;

  // Scratch buffers for the RHS evaluation.
  HeffWorkspace ws;
};

// RK4-compatible RHS callback for Heff evolution.
// Computes: out = dpsi/dt = -i H(t) psi - 0.5 K psi
//
// Preconditions:
// - user != nullptr
// - user->model and user->program are non-null
// - psi.size == out.size == model.hilbert_dim
// - user->ws spans have correct lengths
Status heff_rhs(Real t, Span<const Complex> psi, Span<Complex> out, void* user);

}  // namespace mcwf
}  // namespace fluxpulse