// src/model/effective/duffing_transmon.cpp
#include "fluxpulse/model/effective/duffing_transmon.hpp"

#include <cassert>
#include <utility>

#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/math/real_math.hpp"

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

// a: annihilation operator in truncated basis.
// Elements: <m|a|m+1> = sqrt(m+1)
DenseOp annihilation_a(Dim levels) {
  DenseOp A(levels);
  denseop_set_zero(&A);

  const Dim n = levels;
  const Dim lda = A.lda();
  Complex* data = A.data();
  for (Dim m = 0; m + 1 < n; ++m) {
    const Real v = real_sqrt(static_cast<Real>(m + 1));
    // row=m, col=m+1
    data[m + (m + 1) * lda] = cplx(v, Real{0});
  }
  return A;
}

// a†: creation operator
DenseOp creation_adag(Dim levels) {
  DenseOp A(levels);
  denseop_set_zero(&A);

  const Dim n = levels;
  const Dim lda = A.lda();
  Complex* data = A.data();
  for (Dim m = 0; m + 1 < n; ++m) {
    const Real v = real_sqrt(static_cast<Real>(m + 1));
    // row=m+1, col=m
    data[(m + 1) + m * lda] = cplx(v, Real{0});
  }
  return A;
}

// n: number operator diag(0,1,2,...)
DenseOp number_op_n(Dim levels) {
  DenseOp N(levels);
  denseop_set_zero(&N);

  const Dim lda = N.lda();
  Complex* data = N.data();
  for (Dim m = 0; m < levels; ++m) {
    data[m + m * lda] = cplx(static_cast<Real>(m), Real{0});
  }
  return N;
}

// Build drift diagonal directly: omega*m + 0.5*anharm*m*(m-1)
DenseOp duffing_drift(Dim levels, Real omega, Real anharm) {
  DenseOp H(levels);
  denseop_set_zero(&H);

  const Dim lda = H.lda();
  Complex* data = H.data();
  for (Dim m = 0; m < levels; ++m) {
    const Real mm = static_cast<Real>(m);
    const Real e = omega * mm + Real{0.5} * anharm * mm * (mm - Real{1});
    data[m + m * lda] = cplx(e, Real{0});
  }
  return H;
}

// Hx = (a + a†)
DenseOp drive_x_op(Dim levels) {
  DenseOp a = annihilation_a(levels);
  DenseOp adag = creation_adag(levels);

  // Hx = a + adag (element-wise)
  const Dim n = levels;
  const Dim lda = a.lda();
  Complex* ad = a.data();
  const Complex* bd = adag.data();
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      Complex& z = ad[row + col * lda];
      const Complex w = bd[row + col * lda];
      z.re += w.re;
      z.im += w.im;
    }
  }
  return a;
}

// Hy = i(a† - a)
DenseOp drive_y_op(Dim levels) {
  DenseOp a = annihilation_a(levels);
  DenseOp adag = creation_adag(levels);

  // tmp = adag - a
  const Dim n = levels;
  const Dim lda = a.lda();
  Complex* td = a.data();
  const Complex* bd = adag.data();
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      Complex& z = td[row + col * lda];
      const Complex w = bd[row + col * lda];
      z.re = w.re - z.re;
      z.im = w.im - z.im;
    }
  }

  // Multiply by i: i*(x+iy) = -y + i*x
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      Complex& z = td[row + col * lda];
      const Real xr = z.re;
      const Real xi = z.im;
      z.re = -xi;
      z.im = xr;
    }
  }

  return a;
}

}  // namespace

Status build_duffing_transmon_model(const DuffingTransmonParams& p, ModelBase* out) {
  if (!out) {
    return Error(ErrorCode::kInvalidArgument, "build_duffing_transmon_model: out is null");
  }
  if (p.levels < 2) {
    return Error(ErrorCode::kInvalidArgument, "build_duffing_transmon_model: levels must be >= 2");
  }

  ModelBase m;
  m.hilbert_dim = p.levels;
  m.subsystem_dims = {p.levels};
  m.drift_terms.clear();
  m.control_terms.clear();
  m.collapse_ops.clear();

  // Drift
  {
    DenseOp H0 = duffing_drift(p.levels, p.omega, p.anharm);
    m.drift_terms.push_back(std::move(H0));
  }

  // Controls
  if (p.enable_drive_x) {
    DenseOp Hx = drive_x_op(p.levels);
    denseop_scale_inplace(&Hx, p.drive_scale);

    ControlTerm ct;
    ct.channel = ChannelId::DriveX();
    ct.H = std::move(Hx);
    m.control_terms.push_back(std::move(ct));
  }

  if (p.enable_drive_y) {
    DenseOp Hy = drive_y_op(p.levels);
    denseop_scale_inplace(&Hy, p.drive_scale);

    ControlTerm ct;
    ct.channel = ChannelId::DriveY();
    ct.H = std::move(Hy);
    m.control_terms.push_back(std::move(ct));
  }

  // Noise
  // Relaxation: C = sqrt(gamma1) * a
  if (p.t1 > Real{0}) {
    const Real gamma1 = Real{1} / p.t1;
    DenseOp C = annihilation_a(p.levels);
    denseop_scale_inplace(&C, real_sqrt(gamma1));

    CollapseOp co;
    co.C = std::move(C);
    m.collapse_ops.push_back(std::move(co));
  }

  // Dephasing: C = sqrt(gamma_phi/2) * n
  if (p.tphi > Real{0}) {
    const Real gamma_phi = Real{1} / p.tphi;
    DenseOp C = number_op_n(p.levels);
    denseop_scale_inplace(&C, real_sqrt(gamma_phi * Real{0.5}));

    CollapseOp co;
    co.C = std::move(C);
    m.collapse_ops.push_back(std::move(co));
  }

  Status st = m.validate();
  if (!st.ok()) return Annotate(st, "build_duffing_transmon_model: model.validate");

  *out = std::move(m);
  return Ok();
}

}  // namespace effective
}  // namespace fluxpulse