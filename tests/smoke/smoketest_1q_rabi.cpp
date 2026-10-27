// tests/smoke/smoketest_1q_rabi.cpp
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
  // P1 = |1><1| = diag(0,1)
  fluxpulse::DenseOp P(2);
  for (fluxpulse::Dim col = 0; col < 2; ++col) {
    for (fluxpulse::Dim row = 0; row < 2; ++row) {
      P.data()[row + col * P.lda()] = cplx(fluxpulse::Real{0}, fluxpulse::Real{0});
    }
  }
  P.data()[1 + 1 * P.lda()] = cplx(fluxpulse::Real{1}, fluxpulse::Real{0});
  return P;
}

inline fluxpulse::Real sin2(fluxpulse::Real x) {
  const double s = std::sin(static_cast<double>(x));
  return static_cast<fluxpulse::Real>(s * s);
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
  constexpr Real kTol = Real{0.02};  // smoke-level tolerance

  // ----- Rabi setup -----
  // Hamiltonian: H(t) = (u/2) X for 0 <= t <= T
  // Starting in |0>, P1(t) = sin^2(u t / 2).
  const Real u = Real{10.0};             // Rabi angular frequency in simulation units
  const Real t_pi = kPi / u;             // pi pulse duration
  const Real t_final = Real{0.8};        // long enough to see a couple oscillations
  const Real dt = Real{1e-3};
  const Dim n_points =
      static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;

  // ----- Pulse program: constant drive_x -----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    Envelope env;
    env.add(Segment::PiecewiseConstant(Real{0.0}, t_final, u));

    program.add_pulse(fluxpulse::Pulse(ChannelId::DriveX(), std::move(env)));

    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ----- Model: generic qubit, drift=0, drive_x enabled -----
  ModelBase model;
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{0.0};
    p.detuning = Real{0.0};
    p.t1 = Real{0.0};
    p.tphi = Real{0.0};
    p.enable_drive_x = true;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    const fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &model);
    assert(st.ok());
  }

  // ----- Processor -----
  Processor proc;
  proc.model = std::move(model);
  proc.program = std::move(program);

  proc.observables = {projector_P1()};
  proc.observable_names = {"P1"};

  proc.opt.sample_stride = 1;
  proc.opt.trajectories = 1;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
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

  // ----- Checks -----
  // t=0 => P1 ~ 0
  expect_near(r.expect_at(0, 0), Real{0.0}, kTol);

  auto idx_for_time = [&](Real t) -> Dim {
    const double x = static_cast<double>(t / dt);
    const Dim idx = static_cast<Dim>(std::llround(x));
    return idx;
  };

  // Check around pi pulse: P1 ~ 1
  {
    const Dim idx = idx_for_time(t_pi);
    assert(idx >= 0 && idx < r.num_times());
    const Real t = r.tlist[static_cast<size_t>(idx)];
    const Real expected = sin2(u * t * Real{0.5});
    expect_near(r.expect_at(0, idx), expected, kTol);
    // expected should be close to 1
    assert(expected > Real{0.95});
  }

  // Check around 2*pi pulse: P1 ~ 0
  {
    const Real t_2pi = Real{2} * kPi / u;
    const Dim idx = idx_for_time(t_2pi);
    if (idx >= 0 && idx < r.num_times()) {
      const Real t = r.tlist[static_cast<size_t>(idx)];
      const Real expected = sin2(u * t * Real{0.5});
      expect_near(r.expect_at(0, idx), expected, kTol);
      // expected should be close to 0
      assert(expected < Real{0.05});
    }
  }

  return 0;
}