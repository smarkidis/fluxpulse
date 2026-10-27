// src/sim/mcwf/jump.cpp
#include "fluxpulse/sim/mcwf/jump.hpp"

#include <cmath>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {
namespace mcwf {
namespace {

inline Complex cplx(Real re, Real im) { return Complex{re, im}; }
inline Complex cadd(const Complex& a, const Complex& b) { return cplx(a.re + b.re, a.im + b.im); }
inline Complex cmul(const Complex& a, const Complex& b) {
  return cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
inline bool finite_real(Real x) { return std::isfinite(static_cast<double>(x)) != 0; }

// y = A * x (DenseOp column-major)
Status matvec_denseop(const DenseOp& A, Span<const Complex> x, Span<Complex> y) {
  const Dim n = A.dim();
  if (x.size != n || y.size != n) {
    return Error(ErrorCode::kInvalidArgument, "matvec_denseop: dimension mismatch");
  }

  const Dim lda = A.lda();
  const Complex* M = A.data();
  for (Dim row = 0; row < n; ++row) {
    Complex acc = cplx(Real{0}, Real{0});
    for (Dim col = 0; col < n; ++col) {
      acc = cadd(acc, cmul(M[row + col * lda], x[col]));
    }
    y[row] = acc;
  }
  return Ok();
}

}  // namespace

Status compute_jump_probabilities(Span<const CollapseOp> collapse_ops,
                                 Real dt,
                                 Span<const Complex> psi,
                                 const JumpWorkspace& ws,
                                 Span<Real> p_out,
                                 Real* p_total_out) {
  if (!p_total_out) return Error(ErrorCode::kInvalidArgument, "compute_jump_probabilities: p_total_out is null");
  *p_total_out = Real{0};

  const Dim n_ops = collapse_ops.size;
  if (p_out.size != n_ops) {
    return Error(ErrorCode::kInvalidArgument, "compute_jump_probabilities: p_out size mismatch");
  }

  if (n_ops == 0) {
    for (Dim k = 0; k < p_out.size; ++k) p_out[k] = Real{0};
    return Ok();
  }

  const Dim n = psi.size;
  if (ws.tmp.size != n) {
    return Error(ErrorCode::kInvalidArgument, "compute_jump_probabilities: ws.tmp size mismatch");
  }

  Real total = Real{0};
  for (Dim k = 0; k < n_ops; ++k) {
    const DenseOp& C = collapse_ops[k].C;
    if (C.dim() != n) return Error(ErrorCode::kInvalidArgument, "compute_jump_probabilities: collapse op dim mismatch");

    FLUXPULSE_RETURN_IF_ERROR(matvec_denseop(C, psi, ws.tmp));

    Real n2 = Real{0};
    for (Dim i = 0; i < n; ++i) n2 += ::fluxpulse::abs2(ws.tmp[i]);

    const Real pk = dt * n2;
    p_out[k] = pk;
    total += pk;
  }

  *p_total_out = total;
  return Ok();
}

Status sample_jump_index(Span<const Real> p, Real p_total, Real u01, int* out_index) {
  if (!out_index) return Error(ErrorCode::kInvalidArgument, "sample_jump_index: out_index is null");
  *out_index = -1;

  if (p.size <= 0) return Error(ErrorCode::kInvalidArgument, "sample_jump_index: empty p");
  if (!(p_total > Real{0}) || !finite_real(p_total)) return Error(ErrorCode::kInvalidArgument, "sample_jump_index: p_total invalid");
  if (!(u01 >= Real{0} && u01 < Real{1})) return Error(ErrorCode::kInvalidArgument, "sample_jump_index: u01 must be in [0,1)");

  const Real threshold = u01 * p_total;
  Real acc = Real{0};
  for (Dim k = 0; k < p.size; ++k) {
    if (p[k] < Real{0}) return Error(ErrorCode::kInvalidArgument, "sample_jump_index: negative probability");
    acc += p[k];
    if (threshold < acc) {
      *out_index = static_cast<int>(k);
      return Ok();
    }
  }

  *out_index = static_cast<int>(p.size - 1);
  return Ok();
}

Status normalize_state(Span<Complex> psi_io) {
  Real n2 = Real{0};
  for (Dim i = 0; i < psi_io.size; ++i) n2 += ::fluxpulse::abs2(psi_io[i]);

  if (!(n2 > Real{0}) || !finite_real(n2)) {
    return Error(ErrorCode::kInvalidArgument, "normalize_state: norm is zero or not finite");
  }

  const Real inv = static_cast<Real>(1.0 / std::sqrt(static_cast<double>(n2)));
  for (Dim i = 0; i < psi_io.size; ++i) {
    psi_io[i].re *= inv;
    psi_io[i].im *= inv;
  }
  return Ok();
}

Status apply_jump_and_normalize(const DenseOp& C,
                               Span<Complex> psi_io,
                               const JumpWorkspace& ws,
                               Real* norm2_out) {
  const Dim n = psi_io.size;
  if (C.dim() != n) return Error(ErrorCode::kInvalidArgument, "apply_jump_and_normalize: dim mismatch");
  if (ws.tmp.size != n) return Error(ErrorCode::kInvalidArgument, "apply_jump_and_normalize: ws.tmp size mismatch");

  const Span<const Complex> psi_c(psi_io.data, psi_io.size);
  FLUXPULSE_RETURN_IF_ERROR(matvec_denseop(C, psi_c, ws.tmp));

  Real n2 = Real{0};
  for (Dim i = 0; i < n; ++i) n2 += ::fluxpulse::abs2(ws.tmp[i]);
  if (norm2_out) *norm2_out = n2;

  if (!(n2 > Real{0}) || !finite_real(n2)) {
    return Error(ErrorCode::kInvalidArgument, "apply_jump_and_normalize: post-jump norm is zero or not finite");
  }

  const Real inv = static_cast<Real>(1.0 / std::sqrt(static_cast<double>(n2)));
  for (Dim i = 0; i < n; ++i) {
    psi_io[i].re = ws.tmp[i].re * inv;
    psi_io[i].im = ws.tmp[i].im * inv;
  }
  return Ok();
}

Status mcwf_jump_step(Span<const CollapseOp> collapse_ops,
                      Real dt,
                      Real u_jump,
                      Real u_pick,
                      Span<Complex> psi_io,
                      const JumpWorkspace& ws,
                      Span<Real> p_scratch,
                      bool* did_jump_out,
                      int* jump_index_out,
                      Real* p_total_out) {
  if (!did_jump_out) return Error(ErrorCode::kInvalidArgument, "mcwf_jump_step: did_jump_out is null");
  if (!jump_index_out) return Error(ErrorCode::kInvalidArgument, "mcwf_jump_step: jump_index_out is null");
  if (!p_total_out) return Error(ErrorCode::kInvalidArgument, "mcwf_jump_step: p_total_out is null");

  *did_jump_out = false;
  *jump_index_out = -1;
  *p_total_out = Real{0};

  if (collapse_ops.size == 0) {
    FLUXPULSE_RETURN_IF_ERROR(normalize_state(psi_io));
    return Ok();
  }
  if (p_scratch.size != collapse_ops.size) {
    return Error(ErrorCode::kInvalidArgument, "mcwf_jump_step: p_scratch size mismatch");
  }

  Real p_total = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(
      compute_jump_probabilities(collapse_ops, dt,
                                Span<const Complex>(psi_io.data, psi_io.size),
                                ws, p_scratch, &p_total));
  *p_total_out = p_total;

  if (!jump_occurs(p_total, u_jump)) {
    FLUXPULSE_RETURN_IF_ERROR(normalize_state(psi_io));
    return Ok();
  }

  int idx = -1;
  FLUXPULSE_RETURN_IF_ERROR(sample_jump_index(Span<const Real>(p_scratch.data, p_scratch.size), p_total, u_pick, &idx));
  if (idx < 0 || static_cast<Dim>(idx) >= collapse_ops.size) {
    return Error(ErrorCode::kInternal, "mcwf_jump_step: sampled invalid index");
  }

  FLUXPULSE_RETURN_IF_ERROR(apply_jump_and_normalize(collapse_ops[static_cast<Dim>(idx)].C, psi_io, ws, nullptr));

  *did_jump_out = true;
  *jump_index_out = idx;
  return Ok();
}

}  // namespace mcwf
}  // namespace fluxpulse