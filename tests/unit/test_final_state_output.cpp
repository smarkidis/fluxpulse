// tests/unit/test_final_state_output.cpp
#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/device/processor.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"
#include "fluxpulse/model/effective/generic_qubit.hpp"

namespace {

using fluxpulse::Complex;
using fluxpulse::Dim;
using fluxpulse::ModelBase;
using fluxpulse::Processor;
using fluxpulse::Real;
using fluxpulse::Result;
using fluxpulse::Span;
using fluxpulse::Status;

inline Complex cplx(Real re, Real im = Real{0}) {
  Complex z;
  z.re = re;
  z.im = im;
  return z;
}



inline Real state_norm2(const std::vector<Complex>& psi) {
  Real s = Real{0};
  for (const Complex& z : psi) {
    s += fluxpulse::abs2(z);
  }
  return s;
}

inline void expect_near(Real a, Real b, Real eps) {
  assert(std::fabs(static_cast<double>(a - b)) <= static_cast<double>(eps));
}

fluxpulse::PulseProgram make_idle_program() {
  // No pulses. With omega=detuning=0 and no couplings/noise, the state
  // must remain unchanged. This isolates final-state output plumbing from
  // pulse-calibration details.
  return fluxpulse::PulseProgram(
      fluxpulse::TimeGrid::Uniform(Real{0}, Real{1e-3}, Dim{11}));
}

ModelBase make_1q_ideal_model() {
  fluxpulse::effective::GenericQubitParams p;
  p.omega = Real{0};
  p.detuning = Real{0};
  p.t1 = Real{0};
  p.tphi = Real{0};
  p.enable_drive_x = false;
  p.enable_drive_y = false;
  p.enable_drive_z = false;

  ModelBase model;
  const Status st = fluxpulse::effective::build_generic_qubit_model(p, &model);
  assert(st.ok());
  assert(model.hilbert_dim == 2);
  assert(model.collapse_ops.empty());
  return model;
}

ModelBase make_nq_ideal_model(Dim nqubits) {
  fluxpulse::effective::CoupledQubitsParams p;
  p.nqubits = static_cast<fluxpulse::Index>(nqubits);

  p.omega.assign(static_cast<size_t>(nqubits), Real{0});
  p.detuning.assign(static_cast<size_t>(nqubits), Real{0});
  p.t1.assign(static_cast<size_t>(nqubits), Real{0});
  p.tphi.assign(static_cast<size_t>(nqubits), Real{0});

  p.enable_drive_x = false;
  p.enable_drive_y = false;
  p.enable_drive_z = false;

  // No exchange/ZZ couplings for this output-plumbing test.
  p.exchange_xy.clear();
  p.zz.clear();

  ModelBase model;
  const Status st = fluxpulse::effective::build_coupled_qubits_model(p, &model);
  assert(st.ok());
  assert(model.collapse_ops.empty());
  return model;
}

void check_single_trajectory_final_state(const ModelBase& model,
                                         Dim occupied_index) {
  Processor proc;
  proc.model = model;
  proc.program = make_idle_program();

  // No observables are needed: this test is specifically about state-vector
  // output. The solver must still return the final state.
  proc.observables.clear();
  proc.observable_names.clear();

  proc.opt.trajectories = 1;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
  proc.opt.sample_stride = 1;
  proc.opt.renormalize_each_step = true;
  proc.opt.diagnostic_stride = 1;
  proc.opt.diagnostics = nullptr;

  std::vector<Complex> psi0(static_cast<size_t>(model.hilbert_dim),
                            cplx(Real{0}));
  psi0[static_cast<size_t>(occupied_index)] = cplx(Real{1});

  Result result;
  const Status st = proc.run_mcwf(
      Span<const Complex>(psi0.data(), static_cast<Dim>(psi0.size())),
      &result);
  assert(st.ok());

  // A single trajectory has a well-defined pure final state.
  assert(result.has_final_state());
  assert(result.final_state.size() ==
         static_cast<size_t>(model.hilbert_dim));

  // The idle, noiseless model must preserve the selected basis state.
  for (Dim i = 0; i < model.hilbert_dim; ++i) {
    const Real p = fluxpulse::abs2(result.final_state[static_cast<size_t>(i)]);
    if (i == occupied_index) {
      expect_near(p, Real{1}, Real{1e-12});
    } else {
      expect_near(p, Real{0}, Real{1e-12});
    }
  }

  expect_near(state_norm2(result.final_state), Real{1}, Real{1e-12});
}

void check_multi_trajectory_has_no_final_state() {
  Processor proc;
  proc.model = make_1q_ideal_model();
  proc.program = make_idle_program();
  proc.observables.clear();
  proc.observable_names.clear();

  proc.opt.trajectories = 4;
  proc.opt.base_seed = static_cast<fluxpulse::Seed>(12345u);
  proc.opt.sample_stride = 1;
  proc.opt.renormalize_each_step = true;
  proc.opt.diagnostic_stride = 1;
  proc.opt.diagnostics = nullptr;

  std::vector<Complex> psi0(2, cplx(Real{0}));
  psi0[0] = cplx(Real{1});

  Result result;

  // Deliberately put stale data in the Result. Processor::run_mcwf() must
  // clear it when starting an ensemble run.
  result.final_state.push_back(cplx(Real{99}, Real{99}));

  const Status st = proc.run_mcwf(
      Span<const Complex>(psi0.data(), static_cast<Dim>(psi0.size())),
      &result);
  assert(st.ok());

  // An MCWF ensemble has no unique final state vector.
  assert(!result.has_final_state());
  assert(result.final_state.empty());
}

void check_result_reset_semantics() {
  Result r;

  r.final_state.push_back(cplx(Real{1}));
  r.resize(/*n_observables=*/0, /*n_times=*/2);
  assert(r.final_state.empty());

  r.final_state.push_back(cplx(Real{1}));
  r.clear();
  assert(r.final_state.empty());
}

}  // namespace

int main() {
  // Result lifecycle: stale state vectors must never survive a reset.
  check_result_reset_semantics();

  // QML ideal-state use cases:
  //
  // 1Q: |1> -> state dimension 2.
  check_single_trajectory_final_state(
      make_1q_ideal_model(),
      /*occupied_index=*/1);

  // 2Q little-endian basis: index 2 corresponds to |q1 q0> = |10>.
  {
    ModelBase model = make_nq_ideal_model(/*nqubits=*/2);
    assert(model.hilbert_dim == 4);
    check_single_trajectory_final_state(model, /*occupied_index=*/2);
  }

  // 3Q little-endian basis: index 5 corresponds to |q2 q1 q0> = |101>.
  {
    ModelBase model = make_nq_ideal_model(/*nqubits=*/3);
    assert(model.hilbert_dim == 8);
    check_single_trajectory_final_state(model, /*occupied_index=*/5);
  }

  // Ensemble semantics: no unique state vector.
  check_multi_trajectory_has_no_final_state();

  return 0;
}
