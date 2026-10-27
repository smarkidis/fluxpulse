// tests/smoke/smoketest_3q_zz_chain.cpp
#include "fluxpulse/device/processor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

// Build X on qubit `q` (little-endian: q=0 flips bit 0).
inline fluxpulse::DenseOp X_on_qubit(int q) {
  using fluxpulse::Complex;
  using fluxpulse::DenseOp;
  using fluxpulse::Real;

  DenseOp X(8);
  X.set_zero();

  const int bit = 1 << q;
  for (int i = 0; i < 8; ++i) {
    const int j = i ^ bit;
    if (i < j) {
      X.set(i, j, fluxpulse::complex_one());
      X.set(j, i, fluxpulse::complex_one());
    }
  }
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

  // ---------------------------------------------------------------------------
  // 3-qubit ZZ chain benchmark with a known closed-form result.
  //
  // Hamiltonian:
  //   H = zeta * (Z0 Z1 + Z1 Z2)
  //
  // Initial state:
  //   |+++> = (|0>+|1>)/sqrt(2) ⊗ (|0>+|1>)/sqrt(2) ⊗ (|0>+|1>)/sqrt(2)
  //
  // Expected:
  //   <X0>(t) = cos(2 zeta t)
  //   <X2>(t) = cos(2 zeta t)
  //   <X1>(t) = cos^2(2 zeta t)
  //
  // In particular at t = pi/(2 zeta):
  //   cos(2 zeta t) = cos(pi) = -1
  //   => <X0> = <X2> = -1, <X1> = (+1)
  // ---------------------------------------------------------------------------

  const Real zeta = Real{5.0};

  const Real t_final = Real{0.35};
  const Real dt = Real{1e-3};
  const Dim n_points = static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;

  // Pulse program: no drives.
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // Model: 3 coupled qubits with ZZ chain only.
  ModelBase model;
  {
    fluxpulse::effective::CoupledQubitsParams p;
    p.nqubits = 3;

    p.omega = {Real{0.0}, Real{0.0}, Real{0.0}};
    p.detuning = {Real{0.0}, Real{0.0}, Real{0.0}};
    p.t1 = {Real{0.0}, Real{0.0}, Real{0.0}};
    p.tphi = {Real{0.0}, Real{0.0}, Real{0.0}};

    p.enable_drive_x = false;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    p.zz.push_back({/*q0=*/0, /*q1=*/1, /*strength=*/zeta});
    p.zz.push_back({/*q0=*/1, /*q1=*/2, /*strength=*/zeta});

    const fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &model);
    assert(st.ok());
  }

  // Processor
  Processor proc;
  proc.model = std::move(model);
  proc.program = std::move(program);

  proc.observables = {X_on_qubit(0), X_on_qubit(1), X_on_qubit(2)};
  proc.observable_names = {"X0", "X1", "X2"};

  proc.opt.sample_stride = 1;
  proc.opt.trajectories = 1;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
  proc.opt.renormalize_each_step = true;

  // Initial state |+++> in basis [|q2 q1 q0>] (little-endian indexing).
  const Real inv_sqrt8 = static_cast<Real>(1.0 / std::sqrt(8.0));
  std::vector<Complex> psi0(8);
  for (int i = 0; i < 8; ++i) {
    psi0[i].re = inv_sqrt8;
    psi0[i].im = Real{0};
  }

  Result r;
  {
    const fluxpulse::Status st =
        proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 8), &r);
    assert(st.ok());
  }

  assert(r.num_observables() == 3);
  assert(r.num_times() == n_points);

  auto idx_for_time = [&](Real t) -> Dim {
    const double x = static_cast<double>(t / dt);
    return static_cast<Dim>(std::llround(x));
  };

  // t=0: all <Xk> ~ 1
  expect_near(r.expect_at(0, 0), Real{1.0}, kTol);
  expect_near(r.expect_at(1, 0), Real{1.0}, kTol);
  expect_near(r.expect_at(2, 0), Real{1.0}, kTol);

  // t = pi/(4 zeta): cos(pi/2)=0 => <X0>=<X2>=0 and <X1>=0
  {
    const Real t0 = kPi / (Real{4} * zeta);
    const Dim idx = idx_for_time(t0);
    assert(idx >= 0 && idx < r.num_times());

    const Real t = r.tlist[static_cast<size_t>(idx)];
    const Real c = cos_real(Real{2} * zeta * t);
    const Real expected_x0 = c;
    const Real expected_x2 = c;
    const Real expected_x1 = c * c;

    expect_near(r.expect_at(0, idx), expected_x0, kTol);
    expect_near(r.expect_at(2, idx), expected_x2, kTol);
    expect_near(r.expect_at(1, idx), expected_x1, kTol);
  }

  // t = pi/(2 zeta): cos(pi)=-1 => <X0>=<X2>=-1, <X1>=+1
  {
    const Real t1 = kPi / (Real{2} * zeta);
    const Dim idx = idx_for_time(t1);
    assert(idx >= 0 && idx < r.num_times());

    const Real t = r.tlist[static_cast<size_t>(idx)];
    const Real c = cos_real(Real{2} * zeta * t);
    const Real expected_x0 = c;
    const Real expected_x2 = c;
    const Real expected_x1 = c * c;

    expect_near(r.expect_at(0, idx), expected_x0, kTol);
    expect_near(r.expect_at(2, idx), expected_x2, kTol);
    expect_near(r.expect_at(1, idx), expected_x1, kTol);

    // Strong signature:
    assert(expected_x0 < Real{-0.9});
    assert(expected_x2 < Real{-0.9});
    assert(expected_x1 > Real{0.9});
  }

  return 0;
}
