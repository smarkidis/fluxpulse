// include/fluxpulse/sim/mcwf/jump.hpp
#pragma once

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/model_base.hpp"

namespace fluxpulse {
namespace mcwf {

// MCWF jump utilities.
//
// In MCWF, jump probabilities for collapse operators C_n are:
//
//   p_n = dt * <psi| C_n^† C_n |psi> = dt * || C_n |psi> ||^2
//
// A jump occurs with probability p_total = sum_n p_n.
// If a jump occurs, pick index n with probability p_n / p_total,
// then update:
//
//   |psi> <- C_n |psi> / || C_n |psi> ||
//
// This file provides small, allocation-free helpers for:
// - computing p_n and p_total,
// - sampling which jump occurs (given a uniform random),
// - applying the selected jump and normalizing.

// Scratch buffers for jump computations.
// All spans must have size == dim(psi) == model.hilbert_dim.
struct JumpWorkspace {
  Span<Complex> tmp;  // used for tmp = C * psi
};

// Compute probabilities p_n for each collapse operator and their total.
//
// Inputs:
// - collapse_ops: list of CollapseOp (each contains C matrix)
// - dt: time step
// - psi: current state vector
// - ws.tmp: scratch length dim
//
// Outputs:
// - p_out: length == collapse_ops.size() (each entry is p_n)
// - p_total_out: sum of p_out entries
Status compute_jump_probabilities(Span<const CollapseOp> collapse_ops,
                                 Real dt,
                                 Span<const Complex> psi,
                                 const JumpWorkspace& ws,
                                 Span<Real> p_out,
                                 Real* p_total_out);

// Decide whether a jump occurs given p_total and a uniform random u in [0,1).
// Convention: if p_total >= 1, returns jump=true (caller should reduce dt).
inline bool jump_occurs(Real p_total, Real u01) {
  if (p_total <= Real{0}) return false;
  if (p_total >= Real{1}) return true;
  return u01 < p_total;
}

// Sample a jump index using probabilities p_n (not necessarily normalized).
// Inputs:
// - p: length N, non-negative
// - p_total: sum(p)
// - u01: uniform random in [0,1)
//
// Output:
// - out_index: selected index in [0,N-1]
Status sample_jump_index(Span<const Real> p, Real p_total, Real u01, int* out_index);

// Apply a jump operator C to psi in-place and normalize.
// Uses ws.tmp as scratch.
//
// Inputs:
// - C: collapse operator (dim x dim)
// - ws.tmp: scratch length dim
// - psi_io: state vector updated in-place
//
// Output:
// - norm2_out (optional): squared norm of the unnormalized jumped state ||C psi||^2
Status apply_jump_and_normalize(const DenseOp& C,
                               Span<Complex> psi_io,
                               const JumpWorkspace& ws,
                               Real* norm2_out);

// Normalize a state vector in-place.
// Returns error if norm is zero or not finite.
Status normalize_state(Span<Complex> psi_io);

// A convenience helper used in MCWF loops:
//
// 1) compute p_n and p_total
// 2) decide jump using u_jump
// 3) if jump: pick index using u_pick and apply jump
//    else: just normalize_state(psi_io)
//
// Inputs:
// - collapse_ops, dt, psi_io
// - u_jump: uniform in [0,1) for jump/no-jump
// - u_pick: uniform in [0,1) for selecting which collapse operator (only used if jump)
// - p_scratch: length == collapse_ops.size()
//
// Outputs:
// - did_jump_out: whether a jump occurred
// - jump_index_out: selected jump index if did_jump_out==true, else -1
// - p_total_out: total jump probability
Status mcwf_jump_step(Span<const CollapseOp> collapse_ops,
                      Real dt,
                      Real u_jump,
                      Real u_pick,
                      Span<Complex> psi_io,
                      const JumpWorkspace& ws,
                      Span<Real> p_scratch,
                      bool* did_jump_out,
                      int* jump_index_out,
                      Real* p_total_out);

}  // namespace mcwf
}  // namespace fluxpulse