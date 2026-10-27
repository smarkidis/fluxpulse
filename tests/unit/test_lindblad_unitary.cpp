#include <cassert>
#include <cmath>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/operators/pauli.hpp"
#include "fluxpulse/sim/lindblad/lindblad_solver.hpp"

using namespace fluxpulse;

int main() {
  ModelBase model;
  model.hilbert_dim = 2;
  model.subsystem_dims = {2};
  model.drift_terms = {scaled(pauli_x(), Complex{0.5, 0})};
  PulseProgram program(TimeGrid::Uniform(0, 0.01, 101));
  assert(program.validate(true).ok());
  const std::vector<Complex> ground = {Complex{1, 0}, Complex{0, 0}};
  std::vector<Complex> rho0;
  assert(lindblad::rho_from_psi(Span<const Complex>(ground.data(), 2), &rho0).ok());
  const DenseOp p1 = proj1();
  lindblad::LindbladWorkspace ws;
  assert(ws.init(model).ok());
  lindblad::LindbladOptions opt;
  opt.enforce_trace_one = false;
  Result result;
  assert(lindblad::run_lindblad(model, program, Span<const DenseOp>(&p1, 1),
      Span<const Complex>(rho0.data(), 4), opt, &ws, &result).ok());
  // H = X/2 gives P1(t) = sin^2(t/2). This exercises all RK4 stages.
  for (Dim i = 0; i < result.num_times(); ++i) {
    const Real a = std::sin(result.tlist[static_cast<size_t>(i)] / 2);
    assert(std::fabs(result.expect_at(0, i) - a * a) < 1e-9);
  }

  // Check that stabilization projects both entries of each conjugate pair.
  model.drift_terms.clear();
  assert(ws.init(model).ok());
  program.grid = TimeGrid::Uniform(0, 0.01, 2);
  rho0 = {Complex{0.5, 0}, Complex{0.1, -0.2},
          Complex{0.1001, 0.2002}, Complex{0.5, 0}};
  opt.enforce_hermiticity = true;
  assert(lindblad::run_lindblad(model, program, Span<const DenseOp>(&p1, 1),
      Span<const Complex>(rho0.data(), 4), opt, &ws, &result).ok());
  assert(std::fabs(ws.rho[1].re - 0.10005) < 1e-12);
  assert(std::fabs(ws.rho[1].im + 0.2001) < 1e-12);
  assert(std::fabs(ws.rho[2].re - ws.rho[1].re) < 1e-12);
  assert(std::fabs(ws.rho[2].im + ws.rho[1].im) < 1e-12);
  return 0;
}
