// tests/unit/test_mcwf_vs_lindblad_1q_t1.cpp
#include "fluxpulse/device/processor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/effective/generic_qubit.hpp"
#include "fluxpulse/sim/lindblad/lindblad_solver.hpp"

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
  P.set_zero();
  P.set(1, 1, fluxpulse::complex_one());
  return P;
}

inline fluxpulse::DenseOp pauli_Z() {
  // Z = diag(1,-1)
  fluxpulse::DenseOp Z(2);
  Z.set_zero();
  Z.set(0, 0, fluxpulse::complex_one());
  Z.set(1, 1, cplx(fluxpulse::Real{-1}, fluxpulse::Real{0}));
  return Z;
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

  // ---------------------------------------------------------------------------
  // Compare MCWF ensemble average to deterministic Lindblad evolution
  // for a 1-qubit T1 decay (no drives).
  //
  // Model:
  //   - generic_qubit with amplitude damping only (t1 > 0, tphi = 0)
  //   - drift omega=0
  //
  // Initial state:
  //   |1> (excited)
  //
  // Observable:
  //   P1 = |1><1|
  //
  // Analytic:
  //   <P1>(t) = exp(-t / t1)
  //
  // Checks:
  //   1) Lindblad matches analytic within a tight tolerance
  //   2) MCWF averaged (N trajectories) matches Lindblad within a loose tolerance
  // ---------------------------------------------------------------------------

  constexpr Real kT1 = Real{0.30};
  constexpr Real kTFinal = Real{0.50};
  constexpr Real kDt = Real{1e-3};

  constexpr Dim kStride = 5;           // sample every 5 grid points (and last)
  // For Z, the standard error can reach 1/sqrt(N). At N=2000 this is
  // about 0.022, too close to the 0.03 comparison tolerance below.
  constexpr uint64_t kTraj = 20000;
  constexpr Real kTolAnalytic = Real{5e-3};
  constexpr Real kTolMcwfVsDm = Real{3e-2};

  const Dim n_points = static_cast<Dim>(std::llround(static_cast<double>(kTFinal / kDt))) + 1;

  // ----- Pulse program: no pulses -----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, kDt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ----- Model: generic qubit with T1 only -----
  ModelBase model;
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{0.0};
    p.detuning = Real{0.0};
    p.t1 = kT1;
    p.tphi = Real{0.0};
    p.enable_drive_x = false;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    const fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &model);
    assert(st.ok());
  }

  // Observables
  std::vector<fluxpulse::DenseOp> observables;
  observables.push_back(projector_P1());
  observables.push_back(pauli_Z());

  // ----- Lindblad run -----
  Result dm;
  {
    // rho0 = |1><1|
    std::vector<Complex> psi1(2);
    psi1[0] = cplx(Real{0}, Real{0});
    psi1[1] = cplx(Real{1}, Real{0});

    std::vector<Complex> rho0;
    {
      const fluxpulse::Status st =
          fluxpulse::lindblad::rho_from_psi(fluxpulse::Span<const Complex>(psi1.data(), 2), &rho0);
      assert(st.ok());
    }

    fluxpulse::lindblad::LindbladWorkspace ws;
    {
      const fluxpulse::Status st = ws.init(model);
      assert(st.ok());
    }

    fluxpulse::lindblad::LindbladOptions opt;
    opt.sample_stride = kStride;
    opt.enforce_trace_one = true;
    opt.enforce_hermiticity = false;

    const fluxpulse::Status st =
        fluxpulse::lindblad::run_lindblad(
            model,
            program,
            fluxpulse::Span<const fluxpulse::DenseOp>(observables.data(), static_cast<Dim>(observables.size())),
            fluxpulse::Span<const Complex>(rho0.data(), static_cast<Dim>(rho0.size())),
            opt,
            &ws,
            &dm);
    assert(st.ok());
  }

  // Analytic check for Lindblad: P1(t) = exp(-t/t1)
  {
    Real max_err = Real{0};
    for (Dim i = 0; i < dm.num_times(); ++i) {
      const Real t = dm.tlist[static_cast<size_t>(i)];
      const Real expected = static_cast<Real>(std::exp(-static_cast<double>(t / kT1)));
      const Real got = dm.expect_at(/*obs=*/0, i);  // P1
      const Real err = (got > expected) ? (got - expected) : (expected - got);
      if (err > max_err) max_err = err;
    }
    assert(max_err < kTolAnalytic);
  }

  // ----- MCWF run (averaged) -----
  Result mcwf;
  {
    Processor proc;
    proc.model = model;
    proc.program = program;
    proc.observables = observables;
    proc.observable_names = {"P1", "Z"};

    proc.opt.sample_stride = kStride;
    proc.opt.trajectories = kTraj;
    proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
    proc.opt.renormalize_each_step = true;

    // Initial state |1>
    std::vector<Complex> psi0(2);
    psi0[0] = cplx(Real{0}, Real{0});
    psi0[1] = cplx(Real{1}, Real{0});

    const fluxpulse::Status st =
        proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 2), &mcwf);
    assert(st.ok());
  }

  // Compare MCWF vs Lindblad for all recorded times and both observables
  {
    assert(mcwf.num_observables() == dm.num_observables());
    assert(mcwf.num_times() == dm.num_times());

    Real max_diff = Real{0};
    for (Dim o = 0; o < dm.num_observables(); ++o) {
      for (Dim i = 0; i < dm.num_times(); ++i) {
        const Real a = dm.expect_at(o, i);
        const Real b = mcwf.expect_at(o, i);
        const Real d = (a > b) ? (a - b) : (b - a);
        if (d > max_diff) max_diff = d;
      }
    }

    // For N trajectories, typical MCWF error ~ O(1/sqrt(N)).
    // This is a smoke-level bound meant to catch implementation bugs.
    assert(max_diff < kTolMcwfVsDm);
  }

  // Also sanity: Z(t) ~ 1 - 2 P1(t) for this model (no drives, diagonal rho)
  {
    Real max_rel = Real{0};
    for (Dim i = 0; i < dm.num_times(); ++i) {
      const Real p1 = dm.expect_at(0, i);
      const Real z = dm.expect_at(1, i);
      const Real expected_z = Real{1} - Real{2} * p1;
      const Real err = (z > expected_z) ? (z - expected_z) : (expected_z - z);
      if (err > max_rel) max_rel = err;
    }
    // This should be extremely tight (up to numerical integration error).
    assert(max_rel < Real{1e-3});
  }

  return 0;
}
