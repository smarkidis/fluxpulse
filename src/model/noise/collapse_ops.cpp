// src/model/noise/collapse_ops.cpp
#include "fluxpulse/model/noise/collapse_ops.hpp"

#include <cassert>

#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/math/real_math.hpp"

namespace fluxpulse {
namespace noise {
namespace {

inline Complex cplx(Real re, Real im) {
  Complex z;
  z.re = re;
  z.im = im;
  return z;
}

inline void denseop_set_zero(DenseOp* A) {
  assert(A != nullptr);
  const Dim n = A->dim();
  const Dim lda = A->lda();
  Complex* data = A->data();
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      data[row + col * lda] = cplx(Real{0}, Real{0});
    }
  }
}

inline void denseop_scale_inplace(DenseOp* A, Real s) {
  assert(A != nullptr);
  const Dim n = A->dim();
  const Dim lda = A->lda();
  Complex* data = A->data();
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      Complex& z = data[row + col * lda];
      z.re *= s;
      z.im *= s;
    }
  }
}

}  // namespace

DenseOp qubit_sigma_minus() {
  // |0><1| in basis {|0>, |1>}
  DenseOp A(2);
  denseop_set_zero(&A);

  // element (row=0, col=1) = 1
  A.data()[0 + 1 * A.lda()] = cplx(Real{1}, Real{0});
  return A;
}

DenseOp qubit_sigma_z() {
  // diag(1, -1)
  DenseOp A(2);
  denseop_set_zero(&A);

  A.data()[0 + 0 * A.lda()] = cplx(Real{1}, Real{0});
  A.data()[1 + 1 * A.lda()] = cplx(Real{-1}, Real{0});
  return A;
}

Status qubit_amplitude_damping(Real gamma1, DenseOp* out_C) {
  if (!out_C) {
    return Error(ErrorCode::kInvalidArgument, "qubit_amplitude_damping: out_C is null");
  }
  if (!(gamma1 > Real{0})) {
    return Error(ErrorCode::kInvalidArgument, "qubit_amplitude_damping: gamma1 must be > 0");
  }

  DenseOp C = qubit_sigma_minus();
  const Real s = real_sqrt(gamma1);
  denseop_scale_inplace(&C, s);

  *out_C = std::move(C);
  return Ok();
}

Status qubit_pure_dephasing(Real gamma_phi, DenseOp* out_C) {
  if (!out_C) {
    return Error(ErrorCode::kInvalidArgument, "qubit_pure_dephasing: out_C is null");
  }
  if (!(gamma_phi > Real{0})) {
    return Error(ErrorCode::kInvalidArgument, "qubit_pure_dephasing: gamma_phi must be > 0");
  }

  // Convention: C = sqrt(gamma_phi/2) * sigma_z
  DenseOp C = qubit_sigma_z();
  const Real s = real_sqrt(gamma_phi * Real{0.5});
  denseop_scale_inplace(&C, s);

  *out_C = std::move(C);
  return Ok();
}

}  // namespace noise
}  // namespace fluxpulse