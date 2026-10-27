// tests/smoke/smoketest_1q_ramsey.cpp
#include "fluxpulse/device/processor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/envelope.hpp"
#include "fluxpulse/control/pulse.hpp"
#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/segment.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/effective/generic_qubit.hpp"

namespace {

inline fluxpulse::Complex cplx(fluxpulse::Real re, fluxpulse::Real im) {
  fluxpulse::Complex z;
  z.re = re;
  z.im = im;
  return z;
}

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline fluxpulse::DenseOp projector_P1() {
  // P1 = |1><1|
  fluxpulse::DenseOp P(2);
  for (fluxpulse::Dim col = 0; col < 2; ++col) {
    for (fluxpulse::Dim row = 0; row < 2; ++row) {
      P.data()[row + col * P.lda()] = cplx(fluxpulse::Real{0}, fluxpulse::Real{0});
    }
  }
  P.data()[1 + 1 * P.lda()] = cplx(fluxpulse::Real{1}, fluxpulse::Real{0});
  return P;
}

}  // namespace

int main() {
  using fluxpulse::ChannelId;
  using fluxpulse::Complex;
  using fluxpulse::Dim;
  using fluxpulse::Envelope;
  using fluxpulse::ModelBase;
  using fluxpulse::Processor;
  using fluxpulse::Real;
  using fluxpulse::Result;
  using fluxpulse::Segment;
  using fluxpulse::TimeGrid;

  constexpr Real kPi = static_cast<Real>(3.14159265358979323846264338327950288);
  constexpr Real kTol = Real{0.08};  // Ramsey is sensitive; keep smoke tolerance loose.

  // ---- Ramsey sequence ----
  // Model:
  //   H0 = 0.5 * detuning * Z
  // Pulse sequence:
  //   π/2 pulse about X at t in [0, tp)
  //   idle from tp to tp + T
  //   π/2 pulse about X at t in [tp + T, 2tp + T)
  //
  // Ideal analytic (instant pulses) gives:
  //   P1(T) = cos^2(detuning * T / 2)  (starting from |0>)
  // Two same-sign X rotations give |1> when the idle phase is zero.
  //
  // Our pulses have finite width; we choose strong drive to approximate instant rotations.

  const Real detuning = Real{6.0};  // angular frequency for free precession
  const Real idle_T = Real{0.5};

  const Real drive_amp = Real{120.0};     // strong to approximate instantaneous pulses
  const Real tp = kPi / drive_amp;        // duration of π pulse; π/2 uses tp/2
  const Real t_pihalf = tp * Real{0.5};   // π/2 pulse duration

  const Real t_final = Real{2} * t_pihalf + idle_T;
  const Real dt = Real{1e-4};
  const Dim n_points = static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;

  // ---- Pulse program ----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    Envelope env;
    // First π/2
    env.add(Segment::PiecewiseConstant(Real{0.0}, t_pihalf, drive_amp));
    // Second π/2
    env.add(Segment::PiecewiseConstant(t_pihalf + idle_T, Real{2} * t_pihalf + idle_T, drive_amp));

    program.add_pulse(fluxpulse::Pulse(ChannelId::DriveX(), std::move(env)));
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ---- Model ----
  ModelBase model;
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{0.0};
    p.detuning = detuning;
    p.t1 = Real{0.0};
    p.tphi = Real{0.0};
    p.enable_drive_x = true;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    const fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &model);
    assert(st.ok());
  }

  // ---- Processor ----
  Processor proc;
  proc.model = std::move(model);
  proc.program = std::move(program);

  proc.observables = {projector_P1()};
  proc.observable_names = {"P1"};

  proc.opt.sample_stride = 1;
  proc.opt.trajectories = 1;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(777u);
  proc.opt.renormalize_each_step = true;

  // Initial state |0>
  std::vector<Complex> psi0(2);
  psi0[0] = cplx(Real{1}, Real{0});
  psi0[1] = cplx(Real{0}, Real{0});

  Result r;
  {
    const fluxpulse::Status st =
        proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 2), &r);
    assert(st.ok());
  }

  assert(r.num_observables() == 1);
  assert(r.num_times() == n_points);

  // ---- Checks ----
  // At t=0, P1 ~ 0
  expect_near(r.expect_at(0, 0), Real{0.0}, Real{0.03});

  // After first π/2 (approx), P1 ~ 0.5
  {
    const Dim idx = static_cast<Dim>(std::llround(static_cast<double>(t_pihalf / dt)));
    const Real p1 = r.expect_at(0, idx);
    assert(p1 > Real{0.35} && p1 < Real{0.65});
  }

  // Compare to ideal cos^2(detuning*T/2), allowing for finite pulse width.
  {
    const Dim idx_last = r.num_times() - 1;
    const Real p1_final = r.expect_at(0, idx_last);
    const Real expected = static_cast<Real>(
        std::pow(std::cos(static_cast<double>(detuning * idle_T * Real{0.5})), 2.0));

    // Smoke check: final value in reasonable agreement with ideal.
    expect_near(p1_final, expected, kTol);

    // Additional sanity: should not be out of [0,1]
    assert(p1_final >= Real{-1e-6});
    assert(p1_final <= Real{1.0} + Real{1e-6});
  }

  return 0;
}
