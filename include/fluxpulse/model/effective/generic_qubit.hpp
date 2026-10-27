// include/fluxpulse/model/effective/generic_qubit.hpp
#pragma once

#include <cstdint>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/model/model_base.hpp"

namespace fluxpulse {
namespace effective {

// Generic 2-level qubit effective model (Phase 1).
//
// Drift convention (Phase 1):
//   H0 = 0.5 * (omega + detuning) * Z
//
// Control convention (Phase 1):
//   DriveX channel -> Hx = 0.5 * X
//   DriveY channel -> Hy = 0.5 * Y   (optional, enabled by flags)
//   DriveZ channel -> Hz = 0.5 * Z   (optional, for Z control)
//
// Noise convention (Phase 1):
//   - If t1 > 0: amplitude damping with gamma1 = 1/t1 using C = sqrt(gamma1) * sigma_minus
//   - If tphi > 0: pure dephasing with gamma_phi = 1/tphi using C = sqrt(gamma_phi/2) * sigma_z
//
// This header provides a small builder that returns a populated ModelBase.
// The solver and Processor work with ModelBase directly.
struct GenericQubitParams {
  Real omega = Real{0};
  Real detuning = Real{0};

  // If <= 0, disabled
  Real t1 = Real{0};
  Real tphi = Real{0};

  // Which controls to include in ModelBase::control_terms
  bool enable_drive_x = true;
  bool enable_drive_y = false;
  bool enable_drive_z = false;
};

// Build a generic qubit ModelBase with dim=2.
// Returns Status::ok() on success.
Status build_generic_qubit_model(const GenericQubitParams& p, ModelBase* out);

}  // namespace effective
}  // namespace fluxpulse