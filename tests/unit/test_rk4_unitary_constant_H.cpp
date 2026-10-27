// tests/unit/test_rk4_unitary_constant_H.cpp
#include "fluxpulse/sim/mcwf/rk4.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline fluxpulse::Complex cplx(fluxpulse::Real re, fluxpulse::Real im) {
  return fluxpulse::Complex{re, im};
}

struct ConstXUser {
  fluxpulse::Real omega = fluxpulse::Real{0};  // H = (omega/2) * X
};

// RHS: dpsi/dt = -i * H * psi, with H = (omega/2) X (constant).
fluxpulse::Status rhs_const_x(fluxpulse::Real /*t*/,
                              fluxpulse::Span<const fluxpulse::Complex> psi,
                              fluxpulse::Span<fluxpulse::Complex> out,
                              void* user) {
  if (!user) return fluxpulse::Error(fluxpulse::ErrorCode::kInvalidArgument, "rhs_const_x: user null");
  if (psi.size != 2 || out.size != 2) {
    return fluxpulse::Error(fluxpulse::ErrorCode::kInvalidArgument, "rhs_const_x: dim mismatch");
  }
  const auto* u = static_cast<const ConstXUser*>(user);
  const fluxpulse::Real s = u->omega * fluxpulse::Real{0.5};

  // For psi = [a, b], X psi = [b, a]
  // Multiply by (-i*s): (-i)*(re + i im) = im - i re
  const fluxpulse::Complex b = psi[1];
  const fluxpulse::Complex a = psi[0];

  out[0].re = s * b.im;
  out[0].im = -s * b.re;

  out[1].re = s * a.im;
  out[1].im = -s * a.re;

  return fluxpulse::Ok();
}

}  // namespace

int main() {
  using fluxpulse::Complex;
  using fluxpulse::Real;
  using fluxpulse::Span;

  constexpr Real kEpsAmp = Real{5e-7};
  constexpr Real kEpsNorm = Real{5e-7};

  const Real omega = Real{2.0};  // so omega/2 = 1.0
  const Real T = Real{1.0};
  const Real dt = Real{1e-3};

  const int n_steps = static_cast<int>(std::llround(static_cast<double>(T / dt)));
  expect_near(dt * static_cast<Real>(n_steps), T, Real{1e-12});

  std::vector<Complex> psi(2), psi_next(2);

  // Initial |0>
  psi[0] = cplx(Real{1}, Real{0});
  psi[1] = cplx(Real{0}, Real{0});

  std::vector<Complex> k1(2), k2(2), k3(2), k4(2), tmp(2);
  fluxpulse::mcwf::Rk4Workspace ws;
  ws.k1 = Span<Complex>(k1.data(), 2);
  ws.k2 = Span<Complex>(k2.data(), 2);
  ws.k3 = Span<Complex>(k3.data(), 2);
  ws.k4 = Span<Complex>(k4.data(), 2);
  ws.tmp = Span<Complex>(tmp.data(), 2);

  ConstXUser user;
  user.omega = omega;

  Real t = Real{0};
  for (int step = 0; step < n_steps; ++step) {
    const auto st = fluxpulse::mcwf::rk4_step(
        t, dt,
        Span<const Complex>(psi.data(), 2),
        Span<Complex>(psi_next.data(), 2),
        rhs_const_x,
        &user,
        ws);
    assert(st.ok());

    psi.swap(psi_next);
    t += dt;
  }

  // Analytic: cos(ωt/2)|0> - i sin(ωt/2)|1>
  const Real angle = omega * T * Real{0.5};
  const Real c = static_cast<Real>(std::cos(static_cast<double>(angle)));
  const Real s = static_cast<Real>(std::sin(static_cast<double>(angle)));

  expect_near(psi[0].re, c, kEpsAmp);
  expect_near(psi[0].im, Real{0}, kEpsAmp);
  expect_near(psi[1].re, Real{0}, kEpsAmp);
  expect_near(psi[1].im, -s, kEpsAmp);

  const Real norm2 = psi[0].re * psi[0].re + psi[0].im * psi[0].im +
                     psi[1].re * psi[1].re + psi[1].im * psi[1].im;
  expect_near(norm2, Real{1}, kEpsNorm);

  return 0;
}