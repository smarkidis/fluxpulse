// tests/unit/test_mcwf_t1_decay.cpp
#include "fluxpulse/model/effective/generic_qubit.hpp"
#include "fluxpulse/sim/mcwf/mcwf_solver.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"
#include "fluxpulse/math/complex.hpp"

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

fluxpulse::DenseOp projector_excited_P1() {
  // P1 = |1><1| = diag(0, 1)
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
  using fluxpulse::DenseOp;
  using fluxpulse::Dim;
  using fluxpulse::ModelBase;
  using fluxpulse::PulseProgram;
  using fluxpulse::Real;
  using fluxpulse::Seed;
  using fluxpulse::TrajectoryId;
  using fluxpulse::TimeGrid;

  // ----------------------------
  // 1) Build a qubit model with T1 only (no drives)
  // ----------------------------
  fluxpulse::effective::GenericQubitParams p;
  p.omega = Real{0};
  p.detuning = Real{0};
  p.enable_drive_x = false;
  p.enable_drive_y = false;
  p.enable_drive_z = false;

  const Real T1 = Real{1.0};  // gamma1 = 1/T1 = 1
  p.t1 = T1;
  p.tphi = Real{0};

  ModelBase model;
  {
    const fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &model);
    assert(st.ok());
    assert(model.validate().ok());
    assert(model.hilbert_dim == 2);
    assert(model.collapse_ops.size() == 1);  // only T1 collapse op
  }

  // ----------------------------
  // 2) PulseProgram: no pulses, just a uniform grid
  // ----------------------------
  const Real dt = Real{0.01};
  const Real t_final = Real{2.0};

  // Make sure t_final aligns exactly to dt*(n_points-1)
  const Dim n_points = static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;
  expect_near(dt * static_cast<Real>(n_points - 1), t_final, Real{1e-12});

  PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
    assert(program.pulses.empty());
  }

  // ----------------------------
  // 3) Observable: excited population P1
  // ----------------------------
  std::vector<DenseOp> observables;
  observables.push_back(projector_excited_P1());

  // ----------------------------
  // 4) Initial state: |1>
  // ----------------------------
  std::vector<Complex> psi0(2);
  psi0[0] = cplx(Real{0}, Real{0});
  psi0[1] = cplx(Real{1}, Real{0});

  // ----------------------------
  // 5) MCWF: average many trajectories
  // ----------------------------
  const Seed base_seed = static_cast<Seed>(12345u);
  const Dim n_traj = 2000;

  fluxpulse::mcwf::McwfWorkspace ws;
  {
    const fluxpulse::Status st = ws.init(model, program, /*n_observables=*/1);
    assert(st.ok());
  }

  fluxpulse::mcwf::McwfOptions opt;
  opt.sample_stride = 1;
  opt.renormalize_each_step = true;

  const Dim n_samples = program.grid.size();  // stride=1 => record every point
  std::vector<Real> sum_expect(static_cast<size_t>(n_samples), Real{0});

  for (Dim traj = 0; traj < n_traj; ++traj) {
    const Seed seed = fluxpulse::derive_trajectory_seed(base_seed,
                                                        static_cast<TrajectoryId>(traj));

    fluxpulse::Rng rng;
    rng.init_splitmix64();
    rng.seed(seed);

    const fluxpulse::Status st = fluxpulse::mcwf::run_mcwf_trajectory_accumulate(
        model,
        program,
        fluxpulse::Span<const DenseOp>(observables.data(), 1),
        fluxpulse::Span<const Complex>(psi0.data(), 2),
        &rng,
        opt,
        &ws,
        fluxpulse::Span<Real>(sum_expect.data(), n_samples));
    assert(st.ok());
  }

  // Convert sum -> mean
  std::vector<Real> mean_expect(static_cast<size_t>(n_samples), Real{0});
  for (Dim i = 0; i < n_samples; ++i) {
    mean_expect[static_cast<size_t>(i)] =
        sum_expect[static_cast<size_t>(i)] / static_cast<Real>(n_traj);
  }

  // ----------------------------
  // 6) Check: <P1>(t) ~ exp(-t/T1)
  // ----------------------------
  const Real gamma = Real{1} / T1;

  // Basic bounds
  for (Dim i = 0; i < n_samples; ++i) {
    const Real v = mean_expect[static_cast<size_t>(i)];
    assert(v >= Real{-1e-6});
    assert(v <= Real{1.0} + Real{1e-6});
  }

  // Statistical tolerance: for a Bernoulli-like quantity with N=2000,
  // sigma ~ sqrt(p(1-p)/N) <= ~0.011. Use ~5 sigma + discretization slack.
  const Real tol = Real{0.07};

  auto idx_for_time = [&](Real t) -> Dim {
    // grid is uniform starting at 0
    return static_cast<Dim>(std::llround(static_cast<double>(t / dt)));
  };

  auto check_time = [&](Real t) {
    const Dim idx = idx_for_time(t);
    assert(idx >= 0 && idx < n_samples);
    const Real expected = exp_real(-gamma * t);
    const Real got = mean_expect[static_cast<size_t>(idx)];
    expect_near(got, expected, tol);
  };

  check_time(Real{0.0});
  check_time(Real{0.5});
  check_time(Real{1.0});
  check_time(Real{1.5});
  check_time(Real{2.0});

  // Soft monotonicity on coarse times (allow small statistical bumps).
  {
    const Real bump_tol = Real{0.04};
    Real prev = mean_expect[0];
    for (Real t : {Real{0.5}, Real{1.0}, Real{1.5}, Real{2.0}}) {
      const Dim idx = idx_for_time(t);
      const Real cur = mean_expect[static_cast<size_t>(idx)];
      assert(cur <= prev + bump_tol);
      prev = cur;
    }
  }

  return 0;
}