// include/fluxpulse/operators/pauli.hpp
#pragma once

// Pauli operators and common single-qubit matrices.
//
// This header provides small helpers to construct 2x2 DenseOp matrices for:
// - I, X, Y, Z
// - raising/lowering operators (sigma_plus / sigma_minus)
//
// Storage/layout: DenseOp is column-major (see linalg/dense_op.hpp).
// Therefore, element (row, col) is set via op.set(row, col, value).
//
// CUDA/BLAS note: these are tiny host-side constructors; they do not perform heavy
// computation and do not affect kernel design.

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/linalg/dense_op.hpp"

namespace fluxpulse {

// 2x2 identity
inline DenseOp pauli_i() {
  DenseOp a(2);
  a.set_zero();
  a.set(0, 0, complex_one());
  a.set(1, 1, complex_one());
  return a;
}

// 2x2 Pauli X
inline DenseOp pauli_x() {
  DenseOp a(2);
  a.set_zero();
  a.set(0, 1, complex_one());
  a.set(1, 0, complex_one());
  return a;
}

// 2x2 Pauli Y
inline DenseOp pauli_y() {
  DenseOp a(2);
  a.set_zero();
  // [[0, -i],
  //  [i,  0]]
  a.set(0, 1, neg(complex_i()));
  a.set(1, 0, complex_i());
  return a;
}

// 2x2 Pauli Z
inline DenseOp pauli_z() {
  DenseOp a(2);
  a.set_zero();
  a.set(0, 0, complex_one());
  a.set(1, 1, Complex{-1.0, 0.0});
  return a;
}

// sigma_plus = |1><0|  = [[0, 0],
//                         [1, 0]]
inline DenseOp sigma_plus() {
  DenseOp a(2);
  a.set_zero();
  a.set(1, 0, complex_one());
  return a;
}

// sigma_minus = |0><1| = [[0, 1],
//                         [0, 0]]
inline DenseOp sigma_minus() {
  DenseOp a(2);
  a.set_zero();
  a.set(0, 1, complex_one());
  return a;
}

// Projectors
inline DenseOp proj0() {
  DenseOp a(2);
  a.set_zero();
  a.set(0, 0, complex_one());
  return a;
}

inline DenseOp proj1() {
  DenseOp a(2);
  a.set_zero();
  a.set(1, 1, complex_one());
  return a;
}

// Convenience scaling helpers (returns a new DenseOp).
inline DenseOp scaled(const DenseOp& op, Complex s) {
  DenseOp out(op.dim());
  out.set_zero();
  const Dim n = op.dim();
  const Complex* A = op.data();
  Complex* B = out.data();
  // Both are column-major with lda = n.
  for (Dim col = 0; col < n; ++col) {
    const Index base = static_cast<Index>(col * n);
    for (Dim row = 0; row < n; ++row) {
      const std::size_t idx = as_size_t(base + row);
      B[idx] = mul(A[idx], s);
    }
  }
  return out;
}

inline DenseOp scaled_real(const DenseOp& op, Real s) {
  return scaled(op, Complex{s, 0.0});
}

}  // namespace fluxpulse