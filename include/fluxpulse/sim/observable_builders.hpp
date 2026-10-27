// include/fluxpulse/sim/observable_builders.hpp
#pragma once

#include <string>
#include <vector>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"

namespace fluxpulse {
namespace sim {

// Build a single observable operator from a human-friendly name.
//
// Supported (Phase 1):
// - Identity: "I" / "id" / "identity"
// - Qubit Pauli (single-qubit):
//     "X", "Y", "Z"               (only valid when dims == [2])
// - Qubit Pauli on specific qubit:
//     "X0", "Y1", "Z2"            (case-insensitive)
// - Qubit Pauli products (k-body; k>=2):
//     "ZZ01", "XY01", "XYZ012"    (indices can be written with separators too: "ZZ0_1", "XY(0,1)")
//   Rules:
//     - letters come first (e.g. "ZZ"), then indices
//     - if suffix is digits-only and its length equals number of letters, each digit is an index (e.g. "ZZ01" => [0,1])
//     - otherwise, suffix is parsed as integer tokens (e.g. "ZZ10_11" => [10,11])
// - Single-qubit projectors:
//     "P0", "P1"                  (qubit 0 by default; use "P1_1" or "P1:1" for other qubits)
// - Full-basis projector (computational basis state):
//     "P:10"                      (big-endian bitstring for qubits; leftmost is qubit N-1)
//     "P:0110"                    (for N=4)
//     For dims.size()==1, "P:k" projects onto basis state k in that single subsystem.
//     If dims.size()>1 and token count doesn't match dims.size(), "P:<int>" is treated as linear index.
// - Number operator for single-mode oscillator/transmon:
//     "N" / "n"                   (only valid when dims.size()==1)
//
// Conventions:
// - Subsystems are in little-endian order (subsystem 0 is least significant).
// - For Pauli operators, only subsystems with dim==2 are supported.
//
// Inputs:
// - name: observable name (see above). Leading/trailing whitespace is ignored.
// - subsystem_dims: dimensions per subsystem, ordered little-endian (dims[0] is subsystem 0).
//
// Output:
// - out: DenseOp sized D x D where D = product(subsystem_dims).
Status build_observable(const std::string& name,
                        const std::vector<Dim>& subsystem_dims,
                        DenseOp* out);

// Build a list of observables in the same order as `names`.
Status build_observables(const std::vector<std::string>& names,
                         const std::vector<Dim>& subsystem_dims,
                         std::vector<DenseOp>* out);

}  // namespace sim
}  // namespace fluxpulse
