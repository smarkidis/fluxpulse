// src/sim/mcwf/rk4.cpp
#include "fluxpulse/sim/mcwf/rk4.hpp"

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"

namespace fluxpulse {
namespace mcwf {
namespace {

inline Complex cadd(const Complex& a, const Complex& b) {
  Complex r;
  r.re = a.re + b.re;
  r.im = a.im + b.im;
  return r;
}

inline Complex cscale(const Complex& a, Real s) {
  Complex r;
  r.re = a.re * s;
  r.im = a.im * s;
  return r;
}

inline Status validate_ws(Dim n, const Rk4Workspace& ws) {
  if (n <= 0) return Error(ErrorCode::kInvalidArgument, "rk4_step: state size must be > 0");
  if (!ws.k1.data || ws.k1.size != n) return Error(ErrorCode::kInvalidArgument, "rk4_step: ws.k1 size mismatch");
  if (!ws.k2.data || ws.k2.size != n) return Error(ErrorCode::kInvalidArgument, "rk4_step: ws.k2 size mismatch");
  if (!ws.k3.data || ws.k3.size != n) return Error(ErrorCode::kInvalidArgument, "rk4_step: ws.k3 size mismatch");
  if (!ws.k4.data || ws.k4.size != n) return Error(ErrorCode::kInvalidArgument, "rk4_step: ws.k4 size mismatch");
  if (!ws.tmp.data || ws.tmp.size != n) return Error(ErrorCode::kInvalidArgument, "rk4_step: ws.tmp size mismatch");
  return Ok();
}

}  // namespace

Status rk4_step(Real t,
                Real dt,
                Span<const Complex> psi,
                Span<Complex> psi_out,
                OdeRhsFn rhs,
                void* user,
                const Rk4Workspace& ws) {
  if (!rhs) return Error(ErrorCode::kInvalidArgument, "rk4_step: rhs is null");

  const Dim n = psi.size;
  if (psi_out.size != n) {
    return Error(ErrorCode::kInvalidArgument, "rk4_step: psi_out size mismatch");
  }
  FLUXPULSE_RETURN_IF_ERROR(validate_ws(n, ws));

  const Real half_dt = dt * Real{0.5};
  const Real dt_over_6 = dt * (Real{1} / Real{6});
  const Real dt_over_3 = dt * (Real{1} / Real{3});

  // k1 = f(t, psi)
  FLUXPULSE_RETURN_IF_ERROR(rhs(t, psi, ws.k1, user));

  // tmp = psi + (dt/2) * k1
  for (Dim i = 0; i < n; ++i) {
    ws.tmp[i] = cadd(psi[i], cscale(ws.k1[i], half_dt));
  }
  const Span<const Complex> tmp_c1(ws.tmp.data, ws.tmp.size);
  FLUXPULSE_RETURN_IF_ERROR(rhs(t + half_dt, tmp_c1, ws.k2, user));

  // tmp = psi + (dt/2) * k2
  for (Dim i = 0; i < n; ++i) {
    ws.tmp[i] = cadd(psi[i], cscale(ws.k2[i], half_dt));
  }
  const Span<const Complex> tmp_c2(ws.tmp.data, ws.tmp.size);
  FLUXPULSE_RETURN_IF_ERROR(rhs(t + half_dt, tmp_c2, ws.k3, user));

  // tmp = psi + dt * k3
  for (Dim i = 0; i < n; ++i) {
    ws.tmp[i] = cadd(psi[i], cscale(ws.k3[i], dt));
  }
  const Span<const Complex> tmp_c3(ws.tmp.data, ws.tmp.size);
  FLUXPULSE_RETURN_IF_ERROR(rhs(t + dt, tmp_c3, ws.k4, user));

  // psi_out = psi + (dt/6)*(k1 + 2k2 + 2k3 + k4)
  for (Dim i = 0; i < n; ++i) {
    Complex y = psi[i];
    y = cadd(y, cscale(ws.k1[i], dt_over_6));
    y = cadd(y, cscale(ws.k2[i], dt_over_3));
    y = cadd(y, cscale(ws.k3[i], dt_over_3));
    y = cadd(y, cscale(ws.k4[i], dt_over_6));
    psi_out[i] = y;
  }

  return Ok();
}

}  // namespace mcwf
}  // namespace fluxpulse