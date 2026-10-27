// include/fluxpulse/operators/ladder.hpp
#pragma once

// Ladder (creation/annihilation) operators for finite-dimensional oscillator truncations.
//
// Provides constructors for DenseOp matrices (column-major) of size N x N:
// - annihilation operator a
// - creation operator a^\dagger
// - number operator n = a^\dagger a
//
// Convention (truncated harmonic oscillator basis):
// - Basis states |k>, k = 0..N-1
// - a |k> = sqrt(k) |k-1>
// - a^\dagger |k> = sqrt(k+1) |k+1>
//
// Storage/layout: DenseOp is column-major. Set elements via op.set(row, col, value).
// No heavy computation here; these are host-side constructors.

#include <cassert>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/math/real_math.hpp"
#include "fluxpulse/linalg/dense_op.hpp"

namespace fluxpulse {

// a (annihilation) operator, dimension N >= 1
inline DenseOp annihilation(Dim n) {
  assert(n >= 1);
  DenseOp a(n);
  a.set_zero();
  // a has nonzero elements on subdiagonal: a_{k-1, k} = sqrt(k)
  for (Dim k = 1; k < n; ++k) {
    const Real val = real_sqrt(static_cast<Real>(k));
    a.set(k - 1, k, Complex{val, 0.0});
  }
  return a;
}

// a^\dagger (creation) operator, dimension N >= 1
inline DenseOp creation(Dim n) {
  assert(n >= 1);
  DenseOp ad(n);
  ad.set_zero();
  // a^\dagger has nonzero elements on superdiagonal: a^\dagger_{k+1, k} = sqrt(k+1)
  for (Dim k = 0; k < n - 1; ++k) {
    const Real val = real_sqrt(static_cast<Real>(k + 1));
    ad.set(k + 1, k, Complex{val, 0.0});
  }
  return ad;
}

// Number operator: n = a^\dagger a = diag(0,1,2,...,N-1)
inline DenseOp number_op(Dim n) {
  assert(n >= 1);
  DenseOp num(n);
  num.set_zero();
  for (Dim k = 0; k < n; ++k) {
    num.set(k, k, Complex{static_cast<Real>(k), 0.0});
  }
  return num;
}

// Diagonal projector |k><k|
inline DenseOp projector(Dim n, Dim k) {
  assert(n >= 1);
  assert(k >= 0 && k < n);
  DenseOp p(n);
  p.set_zero();
  p.set(k, k, complex_one());
  return p;
}

}  // namespace fluxpulse