// src/sim/mcwf/heff.cpp
#include "fluxpulse/sim/mcwf/heff.hpp"

#include <utility>

namespace fluxpulse {
namespace mcwf {
namespace {

inline Complex cplx(Real re, Real im) { return Complex{re, im}; }
inline Complex conjc(const Complex& z) { return cplx(z.re, -z.im); }
inline Complex cadd(const Complex& a, const Complex& b) { return cplx(a.re + b.re, a.im + b.im); }
inline Complex cmul(const Complex& a, const Complex& b) {
  return cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
inline Complex cscale(const Complex& a, Real s) { return cplx(a.re * s, a.im * s); }

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

Status coeff_at_time(const PulseProgram& prog,
                     Real t,
                     ChannelId ch,
                     PulseProgram::Cursor* cursor,
                     Real* out_u) {
  if (!out_u) return Error(ErrorCode::kInvalidArgument, "coeff_at_time: out_u is null");
  *out_u = Real{0};

  if (!cursor) {
    *out_u = prog.coeff(ch, t);
    return Ok();
  }

  const int pidx = prog.find_pulse_index(ch);
  if (pidx < 0) {
    *out_u = Real{0};
    return Ok();
  }
  const size_t upidx = static_cast<size_t>(pidx);
  if (upidx >= prog.pulses.size()) {
    return Error(ErrorCode::kInternal, "coeff_at_time: pulse index out of range");
  }
  if (cursor->env.size() != prog.pulses.size()) {
    return Error(ErrorCode::kInvalidArgument, "coeff_at_time: cursor/env size mismatch");
  }

  *out_u = prog.pulses[upidx].envelope.eval_monotonic(t, &cursor->env[upidx]);
  return Ok();
}

}  // namespace

Status build_cdagc_sum(const ModelBase& model, DenseOp* out_K) {
  if (!out_K) return Error(ErrorCode::kInvalidArgument, "build_cdagc_sum: out_K is null");
  const Dim n = model.hilbert_dim;
  if (n <= 0) return Error(ErrorCode::kInvalidArgument, "build_cdagc_sum: invalid dim");

  if (model.collapse_ops.empty()) {
    *out_K = DenseOp();  // dim==0
    return Ok();
  }

  DenseOp K(n);
  // zero
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      K.data()[row + col * K.lda()] = Complex{Real{0}, Real{0}};
    }
  }

  const Dim ldaK = K.lda();
  Complex* Kd = K.data();

  for (const CollapseOp& co : model.collapse_ops) {
    const DenseOp& C = co.C;
    if (C.dim() != n) return Error(ErrorCode::kInvalidArgument, "build_cdagc_sum: collapse op dim mismatch");

    const Dim ldaC = C.lda();
    const Complex* Cd = C.data();

    // K += C† C
    for (Dim i = 0; i < n; ++i) {
      for (Dim j = 0; j < n; ++j) {
        Complex acc = cplx(Real{0}, Real{0});
        for (Dim k = 0; k < n; ++k) {
          const Complex cki = Cd[k + i * ldaC];
          const Complex ckj = Cd[k + j * ldaC];
          acc = cadd(acc, cmul(conjc(cki), ckj));
        }
        Kd[i + j * ldaK] = cadd(Kd[i + j * ldaK], acc);
      }
    }
  }

  *out_K = std::move(K);
  return Ok();
}

Status heff_rhs(Real t, Span<const Complex> psi, Span<Complex> out, void* user) {
  if (!user) return Error(ErrorCode::kInvalidArgument, "heff_rhs: user is null");

  HeffRhsUser* u = static_cast<HeffRhsUser*>(user);
  if (!u->model || !u->program) {
    return Error(ErrorCode::kInvalidArgument, "heff_rhs: model/program is null");
  }

  const ModelBase& model = *u->model;
  const PulseProgram& prog = *u->program;

  const Dim n = model.hilbert_dim;
  if (psi.size != n || out.size != n) {
    return Error(ErrorCode::kInvalidArgument, "heff_rhs: state dimension mismatch");
  }
  if (u->ws.tmp.size != n) {
    return Error(ErrorCode::kInvalidArgument, "heff_rhs: ws.tmp size mismatch");
  }
  if (u->cdagc_sum && u->cdagc_sum->dim() != 0) {
    if (u->cdagc_sum->dim() != n) return Error(ErrorCode::kInvalidArgument, "heff_rhs: K dim mismatch");
    if (u->ws.tmp_k.size != n) return Error(ErrorCode::kInvalidArgument, "heff_rhs: ws.tmp_k size mismatch");
  }

  // Hpsi accumulator -> out
  for (Dim i = 0; i < n; ++i) out[i] = cplx(Real{0}, Real{0});

  // drift
  for (const DenseOp& H : model.drift_terms) {
    FLUXPULSE_RETURN_IF_ERROR(matvec_denseop(H, psi, u->ws.tmp));
    for (Dim i = 0; i < n; ++i) out[i] = cadd(out[i], u->ws.tmp[i]);
  }

  // controls
  for (const ControlTerm& ct : model.control_terms) {
    Real uk = Real{0};
    FLUXPULSE_RETURN_IF_ERROR(coeff_at_time(prog, t, ct.channel, u->program_cursor, &uk));
    if (uk == Real{0}) continue;

    FLUXPULSE_RETURN_IF_ERROR(matvec_denseop(ct.H, psi, u->ws.tmp));
    for (Dim i = 0; i < n; ++i) out[i] = cadd(out[i], cscale(u->ws.tmp[i], uk));
  }

  // out = -i * Hpsi  (a+ib -> b - i a)
  for (Dim i = 0; i < n; ++i) {
    const Complex h = out[i];
    out[i].re = h.im;
    out[i].im = -h.re;
  }

  // out += -0.5 * K psi
  if (u->cdagc_sum && u->cdagc_sum->dim() != 0) {
    FLUXPULSE_RETURN_IF_ERROR(matvec_denseop(*u->cdagc_sum, psi, u->ws.tmp_k));
    for (Dim i = 0; i < n; ++i) out[i] = cadd(out[i], cscale(u->ws.tmp_k[i], Real{-0.5}));
  }

  return Ok();
}

}  // namespace mcwf
}  // namespace fluxpulse