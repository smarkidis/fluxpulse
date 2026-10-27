// tests/unit/test_mcwf_vs_lindblad_2q_swap_noisy.cpp
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

inline fluxpulse::DenseOp projector_basis(fluxpulse::Dim dim, fluxpulse::Dim basis_index) {
  fluxpulse::DenseOp P(dim);
  P.set_zero();
  P.set(static_cast<fluxpulse::Index>(basis_index),
        static_cast<fluxpulse::Index>(basis_index),
        fluxpulse::complex_one());
  return P;
}

inline fluxpulse::DenseOp pauli_Z_local() {
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
  // for a 2-qubit SWAP-like oscillation from exchange coupling (XX + YY)
  // in the presence of T1 decay.
  //
  // Model:
  //   - coupled_qubits with n=2
  //   - exchange_xy coupling between (0,1)
  //   - amplitude damping (T1) on both qubits
  //   - no drives and no ZZ
  //
  // Initial state:
  //   |10> (big-endian, q1=1,q0=0), which oscillates with |01> under exchange.
  //
  // Observables:
  //   P:10, P:01, P:00, Z0, Z1
  //
  // Check:
  //   MCWF averaged matches Lindblad within a loose tolerance.
  // ---------------------------------------------------------------------------

  constexpr Real kJ = Real{2.0};     // exchange strength (larger => faster oscillations)
  constexpr Real kT1 = Real{5.0};    // weak decay relative to oscillations
  constexpr Real kTFinal = Real{1.60};
  constexpr Real kDt = Real{1e-3};

  constexpr Dim kStride = 5;
  constexpr uint64_t kTraj = 2000;
  constexpr Real kTolMcwfVsDm = Real{8e-2};

  const Dim n_points = static_cast<Dim>(
      std::llround(static_cast<double>(kTFinal / kDt))) + 1;

  // ----- Pulse program: no pulses -----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, kDt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ----- Model: 2 qubits, exchange coupling + T1 -----
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

    // Coupling: exchange XY between qubits 0 and 1
    p.exchange_xy.clear();
    p.exchange_xy.push_back({/*q0=*/0, /*q1=*/1, /*strength=*/kJ});

    // No ZZ
    p.zz.clear();

    // Noise
    p.t1 = {kT1, kT1};
    p.tphi.clear();

    const fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &model);
    assert(st.ok());
    assert(model.validate().ok());
  }

  // Observables
  std::vector<fluxpulse::DenseOp> observables;
  // Basis (little-endian): [|00>,|01>,|10>,|11>]
  observables.push_back(projector_basis(4, 2));  // P:10
  observables.push_back(projector_basis(4, 1));  // P:01
  observables.push_back(projector_basis(4, 0));  // P:00
  observables.push_back(Z_qubit(0, 2));          // Z0
  observables.push_back(Z_qubit(1, 2));          // Z1

  // ----- Lindblad run -----
  Result dm;
  {
    // psi0 = |10> => index 2
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

  // ----- MCWF run (averaged) -----
  Result mc;
  {
    Processor proc;
    proc.model = model;
    proc.program = program;
    proc.observables = observables;
    proc.observable_names = {"P:10", "P:01", "P:00", "Z0", "Z1"};

    proc.opt.sample_stride = kStride;
    proc.opt.trajectories = kTraj;
    proc.opt.base_seed = static_cast<fluxpulse::Seed>(24680u);
    proc.opt.renormalize_each_step = true;

    std::vector<Complex> psi0(4);
    psi0[0] = cplx(Real{0}, Real{0});
    psi0[1] = cplx(Real{0}, Real{0});
    psi0[2] = cplx(Real{1}, Real{0});
    psi0[3] = cplx(Real{0}, Real{0});

    const fluxpulse::Status st = proc.run_mcwf(fluxpulse::Span<const Complex>(psi0.data(), 4), &mc);
    assert(st.ok());
  }

  // Compare MCWF vs Lindblad
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

    // For N trajectories, typical MCWF error ~ O(1/sqrt(N)).
    // This is a smoke-level bound meant to catch implementation bugs.
    assert(max_diff < kTolMcwfVsDm);
  }

  // Sanity: probabilities are bounded and total probability should be <= 1.
  // (Not enforcing exact conservation due to noise and finite observable set.)
  {
    for (Dim i = 0; i < dm.num_times(); ++i) {
      const Real p10 = dm.expect_at(0, i);
      const Real p01 = dm.expect_at(1, i);
      const Real p00 = dm.expect_at(2, i);
      assert(p10 >= Real{-1e-6} && p10 <= Real{1} + Real{1e-6});
      assert(p01 >= Real{-1e-6} && p01 <= Real{1} + Real{1e-6});
      assert(p00 >= Real{-1e-6} && p00 <= Real{1} + Real{1e-6});
      const Real s = p10 + p01 + p00;
      assert(s <= Real{1} + Real{5e-3});
    }
  }

  return 0;
}
