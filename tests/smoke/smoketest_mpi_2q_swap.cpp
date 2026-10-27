// tests/smoke/smoketest_mpi_2q_swap.cpp
#include "fluxpulse/device/processor.hpp"
#include "fluxpulse/mpi/mpi_context.hpp"
#include "fluxpulse/mpi/mpi_runner.hpp"
#include "fluxpulse/mpi/mpi_types.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"
#include "fluxpulse/sim/result.hpp"

namespace {

inline fluxpulse::Complex cplx(fluxpulse::Real re, fluxpulse::Real im) {
  fluxpulse::Complex z;
  z.re = re;
  z.im = im;
  return z;
}

inline void expect_near(fluxpulse::Real a,
                        fluxpulse::Real b,
                        fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline fluxpulse::DenseOp projector_basis4(int idx) {
  // |idx><idx| in 4-dim computational basis (little-endian |q1 q0> ordering).
  fluxpulse::DenseOp P(4);
  for (fluxpulse::Dim col = 0; col < 4; ++col) {
    for (fluxpulse::Dim row = 0; row < 4; ++row) {
      P.data()[row + col * P.lda()] =
          cplx(fluxpulse::Real{0}, fluxpulse::Real{0});
    }
  }
  P.data()[idx + idx * P.lda()] =
      cplx(fluxpulse::Real{1}, fluxpulse::Real{0});
  return P;
}

inline fluxpulse::Real sin2(fluxpulse::Real x) {
  const double s = std::sin(static_cast<double>(x));
  return static_cast<fluxpulse::Real>(s * s);
}

}  // namespace

int main(int argc, char** argv) {
  using fluxpulse::Complex;
  using fluxpulse::Dim;
  using fluxpulse::ModelBase;
  using fluxpulse::Processor;
  using fluxpulse::Real;
  using fluxpulse::Result;
  using fluxpulse::TimeGrid;

  fluxpulse::mpi::MpiContext mpi_ctx(argc, argv);
  assert(mpi_ctx.valid());

  constexpr Real kPi =
      static_cast<Real>(3.14159265358979323846264338327950288);
  constexpr Real kTol = Real{0.03};

  // ----- SWAP setup -----
  // H = J (X0 X1 + Y0 Y1)
  // In the single-excitation subspace {|01>,|10>}:
  //   P01(t) = sin^2(2 J t),   P10(t) = cos^2(2 J t)
  const Real J = Real{5.0};
  const Real t_swap = kPi / (Real{4} * J);

  const Real t_final = Real{0.4};
  const Real dt = Real{1e-3};
  const Dim n_points =
      static_cast<Dim>(std::llround(static_cast<double>(t_final / dt))) + 1;

  // ----- Pulse program: no drives -----
  fluxpulse::PulseProgram program(TimeGrid::Uniform(Real{0.0}, dt, n_points));
  {
    const fluxpulse::Status st = program.validate(/*finalize_envelopes=*/true);
    assert(st.ok());
  }

  // ----- Model: exchange coupling only -----
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

    p.exchange_xy.push_back({/*q0=*/0, /*q1=*/1, /*strength=*/J});

    const fluxpulse::Status st =
        fluxpulse::effective::build_coupled_qubits_model(p, &model);
    assert(st.ok());
  }

  // ----- Processor -----
  Processor proc;
  proc.model = std::move(model);
  proc.program = std::move(program);

  // Basis order [|00>,|01>,|10>,|11>] => |01| index 1, |10| index 2.
  proc.observables = {projector_basis4(2), projector_basis4(1)};
  proc.observable_names = {"P10", "P01"};

  proc.opt.sample_stride = 1;
  proc.opt.trajectories = 1;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
  proc.opt.renormalize_each_step = true;
  proc.opt.diagnostics = nullptr;
  proc.opt.diagnostic_stride = 1;

  // Initial state |10>
  std::vector<Complex> psi0(4);
  psi0[0] = cplx(Real{0}, Real{0});
  psi0[1] = cplx(Real{0}, Real{0});
  psi0[2] = cplx(Real{1}, Real{0});
  psi0[3] = cplx(Real{0}, Real{0});

  fluxpulse::mpi::MpiRunInputs in;
  in.processor = &proc;
  in.psi0 =
      fluxpulse::Span<const Complex>(psi0.data(), static_cast<Dim>(psi0.size()));
  in.mpi.total_trajectories =
      static_cast<fluxpulse::TrajectoryId>(proc.opt.trajectories);
  in.mpi.batch_size = 1;
  in.mpi.base_seed = proc.opt.base_seed;
  in.mpi.reproducible = true;
  in.mpi.scheduling = fluxpulse::mpi::SchedulingMode::kStatic;
  in.mpi.enable_trace = false;
  in.mpi.trace_rank = -1;

  fluxpulse::mpi::MpiRunner runner(&mpi_ctx);
  fluxpulse::mpi::MpiRunSummary summary;
  Result r;

  {
    const fluxpulse::Status st =
        runner.Run(in, mpi_ctx.is_root() ? &r : nullptr, &summary);
    assert(st.ok());
  }

  if (!mpi_ctx.is_root()) {
    return 0;
  }

  assert(summary.total_trajectories ==
         static_cast<fluxpulse::TrajectoryId>(proc.opt.trajectories));
  assert(summary.observable_count == 2);
  assert(summary.sample_count == n_points);

  assert(r.num_observables() == 2);
  assert(r.num_times() == n_points);

  auto idx_for_time = [&](Real t) -> Dim {
    return static_cast<Dim>(std::llround(static_cast<double>(t / dt)));
  };

  // t=0: P10 ~ 1, P01 ~ 0
  expect_near(r.expect_at(0, 0), Real{1.0}, kTol);
  expect_near(r.expect_at(1, 0), Real{0.0}, kTol);

  // Near full swap
  {
    const Dim idx = idx_for_time(t_swap);
    assert(idx >= 0 && idx < r.num_times());

    const Real t = r.tlist[static_cast<size_t>(idx)];
    const Real expected_P01 = sin2(Real{2} * J * t);
    const Real expected_P10 = Real{1} - expected_P01;

    expect_near(r.expect_at(1, idx), expected_P01, kTol);
    expect_near(r.expect_at(0, idx), expected_P10, kTol);

    assert(expected_P01 > Real{0.95});
    assert(expected_P10 < Real{0.05});
  }

  // Midpoint check
  {
    const Real t_mid = t_swap * Real{0.5};
    const Dim idx = idx_for_time(t_mid);
    assert(idx >= 0 && idx < r.num_times());

    const Real t = r.tlist[static_cast<size_t>(idx)];
    const Real expected_P01 = sin2(Real{2} * J * t);
    const Real expected_P10 = Real{1} - expected_P01;

    expect_near(r.expect_at(1, idx), expected_P01, kTol);
    expect_near(r.expect_at(0, idx), expected_P10, kTol);
  }

  return 0;
}