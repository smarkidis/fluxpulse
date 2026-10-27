// include/fluxpulse/model/effective/coupled_qubits.hpp
#pragma once

#include <cstdint>
#include <vector>

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/model/model_base.hpp"

namespace fluxpulse {
namespace effective {

// Pair coupling between two qubits (indices in [0, nqubits)).
struct QubitPairCoupling {
  Index q0 = 0;
  Index q1 = 1;

  // Coefficient multiplying the term (see conventions below).
  Real strength = Real{0};
};

// Multi-qubit (2-level per qubit) effective model builder (Phase 1).
//
// Basis / embedding convention:
//   - Little-endian subsystem ordering (qubit 0 is least significant) via operators/tensor.hpp.
//   - |q_{N-1} ... q_1 q_0> -> index = q0 + 2*q1 + 4*q2 + ...
//
// Drift convention (aligned with GenericQubit):
//   H0 = sum_i 0.5 * (omega[i] + detuning[i]) * Z_i
//   - omega/detuning are optional vectors; missing => 0.
//
// Control convention (aligned with GenericQubit):
//   For each enabled axis, per-qubit control term is:
//     DriveX_i -> Hx_i = 0.5 * X_i
//     DriveY_i -> Hy_i = 0.5 * Y_i
//     DriveZ_i -> Hz_i = 0.5 * Z_i
//
//   ChannelId uniqueness rule (ModelBase::validate()):
//     - There must be at most one ControlTerm per ChannelId.
//     - Therefore, multi-qubit control requires distinct ChannelIds per qubit.
//   Default channel ID convention used by this builder when drive_* vectors are empty:
//     DriveX on qubit i: ChannelId{1 + 3*i}
//     DriveY on qubit i: ChannelId{2 + 3*i}
//     DriveZ on qubit i: ChannelId{3 + 3*i}
//   This matches GenericQubit for i=0 (1,2,3) and extends naturally for i>=1.
//
// Coupling conventions:
//   exchange_xy: strength * (X_i X_j + Y_i Y_j)
//   zz:          strength * (Z_i Z_j)
//
// Noise convention (aligned with GenericQubit), per qubit:
//   - If t1[i] > 0: gamma1 = 1/t1[i], C = sqrt(gamma1) * sigma_minus
//   - If tphi[i] > 0: gamma_phi = 1/tphi[i], C = sqrt(gamma_phi/2) * sigma_z
//
// Notes:
// - All terms are embedded into the full Hilbert space (dim = 2^nqubits).
// - For small n (1–4 qubits), dense operators are acceptable in Phase 1.
struct CoupledQubitsParams {
  Index nqubits = 2;

  // Per-qubit drift params (either empty or size == nqubits)
  std::vector<Real> omega;
  std::vector<Real> detuning;

  // Per-qubit noise times (either empty or size == nqubits). <=0 disables per qubit.
  std::vector<Real> t1;
  std::vector<Real> tphi;

  // Enable controls (global flags).
  // If enabled and corresponding drive_* vector is empty, default channels are generated.
  bool enable_drive_x = true;
  bool enable_drive_y = false;
  bool enable_drive_z = false;

  // Optional explicit channels (either empty or size == nqubits).
  // If non-empty: a qubit i contributes a control term only if drive_*[i].valid().
  std::vector<ChannelId> drive_x;
  std::vector<ChannelId> drive_y;
  std::vector<ChannelId> drive_z;

  // Pair couplings
  std::vector<QubitPairCoupling> exchange_xy;
  std::vector<QubitPairCoupling> zz;
};

// Build a coupled-qubits ModelBase with dim = 2^nqubits.
// Returns Status::ok() on success.
Status build_coupled_qubits_model(const CoupledQubitsParams& p, ModelBase* out);

}  // namespace effective
}  // namespace fluxpulse