// include/fluxpulse/model/effective/duffing_transmon.hpp
#pragma once

#include <cstdint>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/model/model_base.hpp"

namespace fluxpulse {
namespace effective {

// Duffing transmon (multi-level) effective model.
//
// Hilbert space: truncated oscillator basis {|0>, |1>, ..., |N-1>}.
//
// Drift Hamiltonian (Phase 1/2 baseline):
//   H0 = omega * n + 0.5 * anharm * n * (n - 1)
// where n is the number operator diag(0,1,2,...).
//
// Controls (optional):
//   DriveX channel: Hx = drive_scale * (a + a†)
//   DriveY channel: Hy = drive_scale * i * (a† - a)
// where a is the annihilation operator in truncated basis.
//
// Noise (optional):
//   - If t1 > 0: relaxation via C = sqrt(gamma1) * a, gamma1 = 1/t1
//   - If tphi > 0: dephasing via C = sqrt(gamma_phi/2) * n, gamma_phi = 1/tphi
//
// Notes:
// - Units are user-defined; keep omega/anharm/drive amplitudes/time consistent.
// - This model is a stepping stone toward more detailed transmon/cQED models.
struct DuffingTransmonParams {
  Dim levels = 0;          // truncation dimension N (>= 2)
  Real omega = Real{0};    // fundamental frequency
  Real anharm = Real{0};   // anharmonicity (often negative for transmons)

  // Drive scaling for (a + a†) and i(a† - a).
  Real drive_scale = Real{1};

  // Optional noise time constants (<=0 disables)
  Real t1 = Real{0};
  Real tphi = Real{0};

  // Control flags
  bool enable_drive_x = true;
  bool enable_drive_y = false;
};

// Build a Duffing transmon ModelBase with dim=levels.
Status build_duffing_transmon_model(const DuffingTransmonParams& p, ModelBase* out);

}  // namespace effective
}  // namespace fluxpulse