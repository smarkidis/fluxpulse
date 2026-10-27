// tests/smoke/smoketest_2q_zz.cpp
#include "fluxpulse/device/processor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"

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

inline fluxpulse::DenseOp X0_full() {
  // X on qubit 0 (least significant): couples |00><01| and |10><11|
  fluxpulse::DenseOp X(4);
  X.set_zero();
  X.set(0, 1, fluxpulse::complex_one());
  X.set(1, 0, fluxpulse::complex_one());
  X.set(2, 3, fluxpulse::complex_one());
  X.set(3, 2, fluxpulse::complex_one());
  return X;
}

inline fluxpulse::DenseOp X1_full() {
  // X on qubit 1 (most significant): couples |00><10| and |01><11|
  fluxpulse::DenseOp X(4);
  X.set_zero();
  X.set(0, 2, fluxpulse::complex_one());
  X.set(2, 0, fluxpulse::complex_one());
  X.set(1, 3, fluxpulse::complex_one());
  X.set(3, 1, fluxpulse::complex_one());
  return X;
}

inline fluxpulse::Real cos_real(fluxpulse::Real x) {
  return static_cast<fluxpulse::Real>(std::cos(static_cast<double>(x)));
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

  constexpr Real kPi = static_cast<Real>(3.14159265358979323846264338327950288);
  constexpr Real kTol = Real{0.03};  // smoke-level tolerance

  // ----- ZZ setup -----
  // Hamiltonian: H = zeta * Z0 Z1
  // Starting in |++> = (|0>+|1>)/sqrt(2) ⊗ (|0>+|1>)/sqrt(2),
  // the reduced single-qubit expectation oscillates:
  //   <X0>(t) = <X1>(t) = cos(2 zeta t)
  const Real zeta = Real{5.0};

  const Real t_final = Real{0.35};
  const Real dt = Real{1e-3};
  const Dim n_points = static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;

  // ----- Pulse program: no drives -----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ----- Model: coupled qubits, ZZ coupling only -----
  ModelBase model;
  {
    fluxpulse::effective::CoupledQubitsParams p;
    p.nqubits = 2;

    p.omega = {Real{0.0}, Real{0.0}};
    p.detuning = {Real{0.0}, Real{0.0}};

    p.t1 = {Real{0.0}, Real{0.0}};
    p.tphi = {Real{0.0}, Real{0.0}};

    p.enable_drive_x = false;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    p.zz.push_back({/*q0=*/0, /*q1=*/1, /*strength=*/zeta});

    const fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &model);
    assert(st.ok());
  }

  // ----- Processor -----
  Processor proc;
  proc.model = std::move(model);
  proc.program = std::move(program);

  proc.observables = {X0_full(), X1_full()};
  proc.observable_names = {"X0", "X1"};

  proc.opt.sample_stride = 1;
  proc.opt.trajectories = 1;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
  proc.opt.renormalize_each_step = true;

  // Initial state |++> in basis [|00>,|01>,|10>,|11>]
  const Real inv2 = Real{0.5};  // 1/2
  std::vector<Complex> psi0(4);
  psi0[0] = cplx(inv2, Real{0});
  psi0[1] = cplx(inv2, Real{0});
  psi0[2] = cplx(inv2, Real{0});
  psi0[3] = cplx(inv2, Real{0});

  Result r;
  {
    const fluxpulse::Status st =
        proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 4), &r);
    assert(st.ok());
  }

  assert(r.num_observables() == 2);
  assert(r.num_times() == n_points);

  auto idx_for_time = [&](Real t) -> Dim {
    const double x = static_cast<double>(t / dt);
    return static_cast<Dim>(std::llround(x));
  };

  // t=0: <X0>=<X1> ~ 1
  expect_near(r.expect_at(0, 0), Real{1.0}, kTol);
  expect_near(r.expect_at(1, 0), Real{1.0}, kTol);

  // t = pi/(4 zeta): cos(pi/2)=0
  {
    const Real t0 = kPi / (Real{4} * zeta);
    const Dim idx = idx_for_time(t0);
    assert(idx >= 0 && idx < r.num_times());
    const Real t = r.tlist[static_cast<size_t>(idx)];
    const Real expected = cos_real(Real{2} * zeta * t);

    expect_near(r.expect_at(0, idx), expected, kTol);
    expect_near(r.expect_at(1, idx), expected, kTol);
    // expected near 0
    assert(std::fabs(static_cast<double>(expected)) < 0.1);
  }

  // t = pi/(2 zeta): cos(pi)=-1
  {
    const Real t1 = kPi / (Real{2} * zeta);
    const Dim idx = idx_for_time(t1);
    if (idx >= 0 && idx < r.num_times()) {
      const Real t = r.tlist[static_cast<size_t>(idx)];
      const Real expected = cos_real(Real{2} * zeta * t);

      expect_near(r.expect_at(0, idx), expected, kTol);
      expect_near(r.expect_at(1, idx), expected, kTol);
      assert(expected < Real{-0.9});
    }
  }

  return 0;
}
