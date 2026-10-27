// include/fluxpulse/model/noise/collapse_ops.hpp
#pragma once

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"

namespace fluxpulse {
namespace noise {

// Helpers to construct common Lindblad collapse operators for Phase 1.
//
// Conventions:
// - Rates are given as gamma (1/time). If you have a time constant T, use gamma = 1/T.
// - All functions return a DenseOp C such that the Lindblad term is:
//     dρ/dt = Σ_n ( C_n ρ C_n† - 0.5 { C_n† C_n, ρ } )
//
// Phase 1 scope:
// - Qubit amplitude damping:   C = sqrt(gamma1) * sigma_minus
// - Qubit pure dephasing:     C = sqrt(gamma_phi/2) * sigma_z
//   (This choice yields the standard master-equation dephasing rate for off-diagonal decay.
//    We can change conventions later if needed, but keep it consistent across code/tests.)

// Returns sigma_minus = |0><1| in the standard {|0>,|1>} basis.
DenseOp qubit_sigma_minus();

// Returns sigma_z = diag(1, -1) in the standard {|0>,|1>} basis.
DenseOp qubit_sigma_z();

// Amplitude damping (T1) for a qubit.
// C = sqrt(gamma1) * sigma_minus
Status qubit_amplitude_damping(Real gamma1, DenseOp* out_C);

// Pure dephasing for a qubit.
// C = sqrt(gamma_phi/2) * sigma_z
//
// Notes:
// - Using sigma_z directly as collapse operator is a valid representation; the factor 1/2
//   sets the coherence decay rate to gamma_phi in the usual convention.
// - If you prefer a different convention, change it once and update tests accordingly.
Status qubit_pure_dephasing(Real gamma_phi, DenseOp* out_C);

}  // namespace noise
}  // namespace fluxpulse