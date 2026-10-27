// tests/unit/test_mcwf_vs_lindblad_2q_t1.cpp
#include "fluxpulse/device/processor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"
#include "fluxpulse/operators/tensor.hpp"
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

inline fluxpulse::DenseOp projector_basis(fluxpulse::Dim dim, fluxpulse::Dim basis_index) {
  // |i><i| in the computational basis.
  fluxpulse::DenseOp P(dim);
  P.set_zero();
  P.set(static_cast<fluxpulse::Index>(basis_index),
        static_cast<fluxpulse::Index>(basis_index),
        fluxpulse::complex_one());
  return P;
}

inline fluxpulse::DenseOp pauli_Z_local() {
  // Z = diag(1,-1)
  fluxpulse::DenseOp Z(2);
  Z.set_zero();
  Z.set(0, 0, fluxpulse::complex_one());
  Z.set(1, 1, cplx(fluxpulse::Real{-1}, fluxpulse::Real{0}));
  return Z;
}

inline fluxpulse::DenseOp Z_qubit(fluxpulse::Index q, fluxpulse::Index nqubits) {
  return fluxpulse::embed_qubit_op(pauli_Z_local(), q, nqubits);
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
  // for a 2-qubit T1 decay (no coupling, no drives).
  //
  // Model:
  //   - coupled_qubits with n=2
  //   - amplitude damping on both qubits (t1 > 0, no tphi)
  //   - no drift detuning/omega and no couplings
  //
  // Initial state:
  //   |10> in big-endian notation (q1=1, q0=0).
  //   Little-endian index: q0 + 2*q1 = 2.
  //
  // Observables:
  //   P:10, P:00, Z0
  //
  // Analytic (for this initial state and pure T1 on qubit1):
  //   <P:10>(t) = exp(-t / t1)
  //   <P:00>(t) = 1 - exp(-t / t1)
  //   <Z0>(t) = +1 (q0 stays in |0>)
  //
  // Checks:
  //   1) Lindblad matches analytic for P:10 and P:00
  //   2) MCWF averaged matches Lindblad within a loose tolerance
  // ---------------------------------------------------------------------------

  constexpr Real kT1 = Real{0.30};
  constexpr Real kTFinal = Real{0.50};
  constexpr Real kDt = Real{1e-3};

  constexpr Dim kStride = 5;          // record every 5 grid points
  constexpr uint64_t kTraj = 2000;    // sampling error ~ O(1/sqrt(N))
  constexpr Real kTolAnalytic = Real{5e-3};
  constexpr Real kTolMcwfVsDm = Real{4e-2};

  const Dim n_points = static_cast<Dim>(
      std::llround(static_cast<double>(kTFinal / kDt))) + 1;

  // ----- Pulse program: no pulses -----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, kDt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ----- Model: 2 qubits, T1 only, no couplings -----
  ModelBase model;
  {
    fluxpulse::effective::CoupledQubitsParams p;
    p.nqubits = 2;
    p.omega = {Real{0}, Real{0}};
    p.detuning = {Real{0}, Real{0}};

    // Disable drives
    p.enable_drive_x = false;
    p.enable_drive_y = false;
    p.enable_drive_z = false;
    p.drive_x.clear();
    p.drive_y.clear();
    p.drive_z.clear();

    // No couplings
    p.exchange_xy.clear();
    p.zz.clear();

    // Noise: T1 on both qubits (Tphi omitted)
    p.t1 = {kT1, kT1};
    p.tphi.clear();

    const fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &model);
    assert(st.ok());
    assert(model.validate().ok());
  }

  // Observables
  std::vector<fluxpulse::DenseOp> observables;
  // Basis ordering (little-endian): |q1 q0> => [|00>,|01>,|10>,|11>]
  // P:10 => index 2, P:00 => index 0
  observables.push_back(projector_basis(/*dim=*/4, /*basis_index=*/2));  // P:10
  observables.push_back(projector_basis(/*dim=*/4, /*basis_index=*/0));  // P:00
  observables.push_back(Z_qubit(/*q=*/0, /*nqubits=*/2));                // Z0

  // ----- Lindblad run -----
  Result dm;
  {
    // psi0 = |10> (big-endian) => little-endian index 2
    std::vector<Complex> psi0(4);
    psi0[0] = cplx(Real{0}, Real{0});
    psi0[1] = cplx(Real{0}, Real{0});
    psi0[2] = cplx(Real{1}, Real{0});
    psi0[3] = cplx(Real{0}, Real{0});

    std::vector<Complex> rho0;
    {
      const fluxpulse::Status st =
          fluxpulse::lindblad::rho_from_psi(fluxpulse::Span<const Complex>(psi0.data(), 4), &rho0);
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

  // Analytic checks for Lindblad
  {
    Real max_err_p10 = Real{0};
    Real max_err_p00 = Real{0};
    Real max_err_z0 = Real{0};

    for (Dim i = 0; i < dm.num_times(); ++i) {
      const Real t = dm.tlist[static_cast<size_t>(i)];
      const Real p10_expected = static_cast<Real>(std::exp(-static_cast<double>(t / kT1)));
      const Real p00_expected = Real{1} - p10_expected;
      const Real z0_expected = Real{1};

      const Real p10 = dm.expect_at(/*obs=*/0, i);
      const Real p00 = dm.expect_at(/*obs=*/1, i);
      const Real z0 = dm.expect_at(/*obs=*/2, i);

      const Real e10 = (p10 > p10_expected) ? (p10 - p10_expected) : (p10_expected - p10);
      const Real e00 = (p00 > p00_expected) ? (p00 - p00_expected) : (p00_expected - p00);
      const Real ez0 = (z0 > z0_expected) ? (z0 - z0_expected) : (z0_expected - z0);

      if (e10 > max_err_p10) max_err_p10 = e10;
      if (e00 > max_err_p00) max_err_p00 = e00;
      if (ez0 > max_err_z0) max_err_z0 = ez0;
    }

    assert(max_err_p10 < kTolAnalytic);
    assert(max_err_p00 < kTolAnalytic);
    assert(max_err_z0 < Real{1e-6});
  }

  // ----- MCWF run (averaged) -----
  Result mc;
  {
    Processor proc;
    proc.model = model;
    proc.program = program;
    proc.observables = observables;
    proc.observable_names = {"P:10", "P:00", "Z0"};

    proc.opt.sample_stride = kStride;
    proc.opt.trajectories = kTraj;
    proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
    proc.opt.renormalize_each_step = true;

    // Initial |10>
    std::vector<Complex> psi0(4);
    psi0[0] = cplx(Real{0}, Real{0});
    psi0[1] = cplx(Real{0}, Real{0});
    psi0[2] = cplx(Real{1}, Real{0});
    psi0[3] = cplx(Real{0}, Real{0});

    const fluxpulse::Status st = proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 4), &mc);
    assert(st.ok());
  }

  // Compare MCWF vs Lindblad for all recorded times and all observables
  {
    assert(mc.num_observables() == dm.num_observables());
    assert(mc.num_times() == dm.num_times());

    Real max_diff = Real{0};
    for (Dim o = 0; o < dm.num_observables(); ++o) {
      for (Dim i = 0; i < dm.num_times(); ++i) {
        const Real a = dm.expect_at(o, i);
        const Real b = mc.expect_at(o, i);
        const Real d = (a > b) ? (a - b) : (b - a);
        if (d > max_diff) max_diff = d;
      }
    }

    // Smoke-level bound meant to catch implementation bugs.
    assert(max_diff < kTolMcwfVsDm);
  }

  // Extra sanity: P:10 + P:00 ~ 1 (only qubit1 decays and q0 stays in |0>)
  {
    Real max_err = Real{0};
    for (Dim i = 0; i < dm.num_times(); ++i) {
      const Real s = dm.expect_at(0, i) + dm.expect_at(1, i);
      const Real err = (s > Real{1}) ? (s - Real{1}) : (Real{1} - s);
      if (err > max_err) max_err = err;
    }
    assert(max_err < Real{1e-3});
  }

  return 0;
}
