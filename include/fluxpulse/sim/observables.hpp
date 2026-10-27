// include/fluxpulse/sim/observables.hpp
#pragma once

#include <vector>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {

// Utilities for expectation values and simple observable collections.
//
// Phase 1 focuses on state-vector simulation (unitary + MCWF trajectories).
// Expectation values are computed as:
//   <O> = <psi| O |psi>
// and we return the real part by default (observables are assumed Hermitian).
//
// Performance:
// - No allocations in hot path: caller provides scratch buffers where needed.
// - Works with DenseOp and spans.

struct Observable {
  std::string name;  // optional label (used for output)
  DenseOp op;        // operator matrix (dim x dim)
};

// Compute tmp = O * psi (matrix-vector product).
// Preconditions:
// - O.dim() == dim(psi) == dim(tmp)
// - psi.size == tmp.size == O.dim()
Status apply_observable(const DenseOp& O, Span<const Complex> psi, Span<Complex> tmp);

// Compute <psi|O|psi>, returning Complex (no Hermitian assumption).
// Uses `tmp` as scratch for O*psi.
//
// Preconditions:
// - O.dim() == psi.size == tmp.size
Status expect_complex(const DenseOp& O, Span<const Complex> psi, Span<Complex> tmp, Complex* out);

// Compute Re(<psi|O|psi>).
// Intended for Hermitian O.
// Uses `tmp` as scratch.
Status expect_real(const DenseOp& O, Span<const Complex> psi, Span<Complex> tmp, Real* out);

// Batched version: compute expectations for multiple observables.
// `tmp` must be length dim and is reused for each observable.
// `out` must have length observables.size().
Status expect_many_real(Span<const DenseOp> observables,
                        Span<const Complex> psi,
                        Span<Complex> tmp,
                        Span<Real> out);

}  // namespace fluxpulse