// include/fluxpulse/operators/tensor.hpp
#pragma once

// Tensor (Kronecker) products and embedding of local operators into a larger Hilbert space.
//
// Motivation:
// - Phase 1 uses dense state vectors and dense operators (small N, e.g. 1–4 qubits).
// - We need a clear, deterministic convention for basis ordering and embedding.
// - Keep solver interfaces simple: solvers only need Op::apply(out,in).
//
// Conventions:
// - Subsystems are ordered with index 0 as the **least significant** (fastest-varying) digit.
//   For qubits, the computational basis ordering is:
//       |q_{N-1} ... q_1 q_0⟩  <->  index = q0 + 2*q1 + 4*q2 + ...
// - Kronecker product kron(A,B) corresponds to A acting on the "left" (more significant)
//   subsystem(s) and B acting on the "right" (less significant) subsystem(s).
//
// Storage:
// - DenseOp uses column-major storage.
//
// Performance:
// - These builders are O(D * d_local) or O(D^2) and intended for small dimensions.
// - For large systems, you would switch to structured/sparse operators or tensor networks.

#include <vector>
#include <algorithm>
#include <cassert>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/linalg/dense_op.hpp"

namespace fluxpulse {

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

inline Dim prod_dims(const std::vector<Dim>& dims) {
  Dim p = 1;
  for (Dim d : dims) {
    assert(d > 0);
    p *= d;
  }
  return p;
}

inline std::vector<Dim> make_strides(const std::vector<Dim>& dims) {
  std::vector<Dim> s(dims.size(), 1);
  for (std::size_t i = 1; i < dims.size(); ++i) {
    s[i] = s[i - 1] * dims[i - 1];
  }
  return s;
}

// 1x1 identity (scalar 1).
inline DenseOp identity1() {
  DenseOp a(1);
  a.set_zero();
  a.set(0, 0, complex_one());
  return a;
}

// dxd identity.
inline DenseOp identity_op(Dim d) {
  DenseOp a(d);
  a.set_identity();
  return a;
}

// -----------------------------------------------------------------------------
// Kronecker product
// -----------------------------------------------------------------------------

// C = kron(A,B) = A ⊗ B
// If A is m×m and B is n×n, then C is (m*n)×(m*n).
// Basis ordering: index = iB + n*iA (B is "right"/least-significant).
inline DenseOp kron(const DenseOp& A, const DenseOp& B) {
  const Dim m = A.dim();
  const Dim n = B.dim();
  assert(m >= 1);
  assert(n >= 1);

  DenseOp C(m * n);
  C.set_zero();

  for (Dim jA = 0; jA < m; ++jA) {
    for (Dim jB = 0; jB < n; ++jB) {
      const Index col = static_cast<Index>(jB + n * jA);
      for (Dim iA = 0; iA < m; ++iA) {
        for (Dim iB = 0; iB < n; ++iB) {
          const Index row = static_cast<Index>(iB + n * iA);
          const Complex v = mul(A.at(iA, jA), B.at(iB, jB));
          C.set(row, col, v);
        }
      }
    }
  }
  return C;
}

// -----------------------------------------------------------------------------
// Embedding: local operator -> full operator
// -----------------------------------------------------------------------------

inline DenseOp embed_local(const DenseOp& local,
                           const std::vector<Dim>& dims,
                           const std::vector<Index>& targets_in) {
  const std::size_t M = dims.size();
  assert(M >= 1);

  std::vector<Index> targets = targets_in;
  assert(!targets.empty());
  std::sort(targets.begin(), targets.end());
  targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
  assert(targets.size() == targets_in.size());
  for (Index t : targets) {
    assert(t >= 0);
    assert(static_cast<std::size_t>(t) < M);
    (void)t;  // silence -Wunused-variable when asserts compile out
  }

  (void)M;  // silence -Wunused-variable when asserts compile out

  const std::vector<Dim> stride = make_strides(dims);
  const Dim D = prod_dims(dims);

  std::vector<Dim> local_dims;
  local_dims.reserve(targets.size());
  for (Index t : targets) local_dims.push_back(dims[static_cast<std::size_t>(t)]);
  const std::vector<Dim> lstride = make_strides(local_dims);
  const Dim d_local = prod_dims(local_dims);

  assert(local.dim() == d_local);

  DenseOp full(D);
  full.set_zero();

  std::vector<Dim> digit_old(targets.size(), 0);
  for (Dim col = 0; col < D; ++col) {
    Dim lc = 0;
    for (std::size_t k = 0; k < targets.size(); ++k) {
      const std::size_t ti = static_cast<std::size_t>(targets[k]);
      const Dim dig = (col / stride[ti]) % dims[ti];
      digit_old[k] = dig;
      lc += dig * lstride[k];
    }

    for (Dim lr = 0; lr < d_local; ++lr) {
      Index row = static_cast<Index>(col);
      for (std::size_t k = 0; k < targets.size(); ++k) {
        const std::size_t ti = static_cast<std::size_t>(targets[k]);
        const Dim dig_new = (lr / lstride[k]) % dims[ti];
        const Dim dig_old = digit_old[k];
        if (dig_new != dig_old) {
          row += static_cast<Index>(dig_new - dig_old) * static_cast<Index>(stride[ti]);
        }
      }
      full.set(row, col, local.at(lr, lc));
    }
  }

  return full;
}

// Convenience helper: embed a product op_a(on qa) ⊗ op_b(on qb) into nqubits.
// Handles little-endian ordering correctly.
inline DenseOp embed_two_qubit_product(const DenseOp& op_a,
                                      const DenseOp& op_b,
                                      Index qa,
                                      Index qb,
                                      Index nqubits) {
  assert(op_a.dim() == 2);
  assert(op_b.dim() == 2);
  assert(nqubits >= 2);
  assert(qa >= 0 && qb >= 0);
  assert(qa < nqubits && qb < nqubits);
  assert(qa != qb);

  Index lo = qa;
  Index hi = qb;
  const DenseOp* op_lo = &op_a;  // acts on less-significant subsystem
  const DenseOp* op_hi = &op_b;  // acts on more-significant subsystem
  if (qa > qb) {
    lo = qb;
    hi = qa;
    op_lo = &op_b;
    op_hi = &op_a;
  }

  // kron(A,B): A acts on the more-significant subsystem, B on the less-significant.
  DenseOp local4 = kron(*op_hi, *op_lo);

  std::vector<Dim> dims(static_cast<std::size_t>(nqubits), 2);
  std::vector<Index> targets{lo, hi};
  return embed_local(local4, dims, targets);
}

inline DenseOp embed_single(const DenseOp& local,
                            const std::vector<Dim>& dims,
                            Index target) {
  std::vector<Index> t{target};
  return embed_local(local, dims, t);
}

inline DenseOp embed_qubit_op(const DenseOp& local2, Index qubit, Index nqubits) {
  assert(nqubits >= 1);
  assert(local2.dim() == 2);
  std::vector<Dim> dims(static_cast<std::size_t>(nqubits), 2);
  return embed_single(local2, dims, qubit);
}

inline DenseOp embed_two_qubit_op(const DenseOp& local4,
                                 Index q0,
                                 Index q1,
                                 Index nqubits) {
  assert(nqubits >= 2);
  assert(local4.dim() == 4);
  if (q0 > q1) { const Index tmp = q0; q0 = q1; q1 = tmp; }
  std::vector<Dim> dims(static_cast<std::size_t>(nqubits), 2);
  std::vector<Index> targets{q0, q1};
  return embed_local(local4, dims, targets);
}

}  // namespace fluxpulse
