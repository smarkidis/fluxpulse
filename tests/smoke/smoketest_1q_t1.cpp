// tests/smoke/smoketest_1q_t1.cpp
#include "fluxpulse/device/processor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
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

inline fluxpulse::Real exp_real(fluxpulse::Real x) {
  return static_cast<fluxpulse::Real>(std::exp(static_cast<double>(x)));
}

}  // namespace

int main() {
  using fluxpulse::Complex;
  using fluxpulse::Dim;
  using fluxpulse::ModelBase;
  using fluxpulse::Processor;
  using fluxpulse::Real;
  using fluxpulse::Result;
  using fluxpulse::TimeGrid;

  constexpr Real kTol = Real{0.07};

  // ---- Model with T1 only ----
  const Real T1 = Real{1.0};
  fluxpulse::effective::GenericQubitParams p;
  p.omega = Real{0.0};
  p.detuning = Real{0.0};
  p.enable_drive_x = false;
  p.enable_drive_y = false;
  p.enable_drive_z = false;
  p.t1 = T1;
  p.tphi = Real{0.0};

  ModelBase model;
  {
    const fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &model);
    assert(st.ok());
    assert(model.collapse_ops.size() == 1);
  }

  // ---- Program: no pulses, uniform time grid ----
  const Real dt = Real{0.01};
  const Real t_final = Real{2.0};
  const Dim n_points = static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;

  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ---- Processor ----
  Processor proc;
  proc.model = std::move(model);
  proc.program = std::move(program);

  proc.observables = {projector_P1()};
  proc.observable_names = {"P1"};

  proc.opt.sample_stride = 1;
  proc.opt.trajectories = 2000;  // enough to reduce Monte Carlo noise
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(42u);
  proc.opt.renormalize_each_step = true;

  // Initial state |1>
  std::vector<Complex> psi0(2);
  psi0[0] = cplx(Real{0}, Real{0});
  psi0[1] = cplx(Real{1}, Real{0});

  Result r;
  {
    const fluxpulse::Status st =
        proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 2), &r);
    assert(st.ok());
  }

  assert(r.num_observables() == 1);
  assert(r.num_times() == n_points);

  const Real gamma = Real{1} / T1;

  auto idx_for_time = [&](Real t) -> Dim {
    return static_cast<Dim>(std::llround(static_cast<double>(t / dt)));
  };

  auto check_time = [&](Real t) {
    const Dim idx = idx_for_time(t);
    assert(idx >= 0 && idx < r.num_times());
    const Real expected = exp_real(-gamma * t);
    const Real got = r.expect_at(0, idx);
    expect_near(got, expected, kTol);
  };

  // t=0: P1 ~ 1
  check_time(Real{0.0});
  // Compare a few anchor points
  check_time(Real{0.5});
  check_time(Real{1.0});
  check_time(Real{1.5});
  check_time(Real{2.0});

  // Coarse monotonic sanity (allow slight Monte Carlo bumps)
  {
    const Real bump_tol = Real{0.04};
    Real prev = r.expect_at(0, 0);
    for (Real t : {Real{0.5}, Real{1.0}, Real{1.5}, Real{2.0}}) {
      const Dim idx = idx_for_time(t);
      const Real cur = r.expect_at(0, idx);
      assert(cur <= prev + bump_tol);
      prev = cur;
    }
  }

  return 0;
}