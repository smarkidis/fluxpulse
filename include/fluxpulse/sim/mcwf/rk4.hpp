// include/fluxpulse/sim/mcwf/rk4.hpp
#pragma once

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {
namespace mcwf {

// ODE callback signature for state-vector evolution.
// Computes out = f(t, psi).
//
// Requirements:
// - psi.size == out.size
// - no allocations in this function (hot path)
using OdeRhsFn = Status (*)(Real t, Span<const Complex> psi, Span<Complex> out, void* user);

// Scratch buffers required by RK4.
// All spans must have the same size as the state vector.
struct Rk4Workspace {
  Span<Complex> k1;
  Span<Complex> k2;
  Span<Complex> k3;
  Span<Complex> k4;
  Span<Complex> tmp;  // used for intermediate states: psi + ...
};

// Single RK4 step:
//   psi_out = psi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
// where
//   k1 = f(t, psi)
//   k2 = f(t + dt/2, psi + dt/2*k1)
//   k3 = f(t + dt/2, psi + dt/2*k2)
//   k4 = f(t + dt,   psi + dt*k3)
//
// Preconditions:
// - psi, psi_out same length
// - workspace buffers same length
// - rhs != nullptr
Status rk4_step(Real t,
                Real dt,
                Span<const Complex> psi,
                Span<Complex> psi_out,
                OdeRhsFn rhs,
                void* user,
                const Rk4Workspace& ws);

}  // namespace mcwf
}  // namespace fluxpulse