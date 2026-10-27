// src/model/effective/generic_qubit.cpp
#include "fluxpulse/model/effective/generic_qubit.hpp"

#include <cassert>
#include <utility>

#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/noise/collapse_ops.hpp"

namespace fluxpulse {
namespace effective {
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

DenseOp pauli_x() {
  // X = [[0,1],[1,0]]
  DenseOp A(2);
  denseop_set_zero(&A);
  A.data()[0 + 1 * A.lda()] = cplx(Real{1}, Real{0});
  A.data()[1 + 0 * A.lda()] = cplx(Real{1}, Real{0});
  return A;
}

DenseOp pauli_y() {
  // Y = [[0,-i],[i,0]]
  DenseOp A(2);
  denseop_set_zero(&A);
  A.data()[0 + 1 * A.lda()] = cplx(Real{0}, Real{-1});  // -i
  A.data()[1 + 0 * A.lda()] = cplx(Real{0}, Real{+1});  // +i
  return A;
}

DenseOp pauli_z() {
  // Z = [[1,0],[0,-1]]
  DenseOp A(2);
  denseop_set_zero(&A);
  A.data()[0 + 0 * A.lda()] = cplx(Real{1}, Real{0});
  A.data()[1 + 1 * A.lda()] = cplx(Real{-1}, Real{0});
  return A;
}

}  // namespace

Status build_generic_qubit_model(const GenericQubitParams& p, ModelBase* out) {
  if (!out) {
    return Error(ErrorCode::kInvalidArgument, "build_generic_qubit_model: out is null");
  }

  ModelBase m;
  m.hilbert_dim = 2;
  m.subsystem_dims = {2};
  m.drift_terms.clear();
  m.control_terms.clear();
  m.collapse_ops.clear();

  // Drift: H0 = 0.5 * (omega + detuning) * Z
  {
    const Real w = p.omega + p.detuning;
    DenseOp H0 = pauli_z();
    denseop_scale_inplace(&H0, Real{0.5} * w);
    m.drift_terms.push_back(std::move(H0));
  }

  // Controls: Hx/Hy/Hz each with a 0.5 factor.
  if (p.enable_drive_x) {
    DenseOp Hx = pauli_x();
    denseop_scale_inplace(&Hx, Real{0.5});
    ControlTerm ct;
    ct.channel = ChannelId::DriveX();
    ct.H = std::move(Hx);
    m.control_terms.push_back(std::move(ct));
  }

  if (p.enable_drive_y) {
    DenseOp Hy = pauli_y();
    denseop_scale_inplace(&Hy, Real{0.5});
    ControlTerm ct;
    ct.channel = ChannelId::DriveY();
    ct.H = std::move(Hy);
    m.control_terms.push_back(std::move(ct));
  }

  if (p.enable_drive_z) {
    DenseOp Hz = pauli_z();
    denseop_scale_inplace(&Hz, Real{0.5});
    ControlTerm ct;
    ct.channel = ChannelId::DriveZ();
    ct.H = std::move(Hz);
    m.control_terms.push_back(std::move(ct));
  }

  // Noise: add collapse operators if enabled.
  if (p.t1 > Real{0}) {
    const Real gamma1 = Real{1} / p.t1;
    DenseOp C;
    Status st = noise::qubit_amplitude_damping(gamma1, &C);
    if (!st.ok()) return Annotate(st, "build_generic_qubit_model: amplitude damping");
    CollapseOp co;
    co.C = std::move(C);
    m.collapse_ops.push_back(std::move(co));
  }

  if (p.tphi > Real{0}) {
    const Real gamma_phi = Real{1} / p.tphi;
    DenseOp C;
    Status st = noise::qubit_pure_dephasing(gamma_phi, &C);
    if (!st.ok()) return Annotate(st, "build_generic_qubit_model: pure dephasing");
    CollapseOp co;
    co.C = std::move(C);
    m.collapse_ops.push_back(std::move(co));
  }

  // Final validation
  Status st = m.validate();
  if (!st.ok()) return Annotate(st, "build_generic_qubit_model: model.validate");

  *out = std::move(m);
  return Ok();
}

}  // namespace effective
}  // namespace fluxpulse