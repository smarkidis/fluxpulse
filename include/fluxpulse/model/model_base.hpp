// include/fluxpulse/model/model_base.hpp
#pragma once

#include <cstdint>
#include <vector>

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/linalg/dense_op.hpp"

namespace fluxpulse {

// A control Hamiltonian term H_k associated with a classical control channel u_k(t).
// The full Hamiltonian is typically:
//   H(t) = sum_j H_drift[j] + sum_k u_k(t) * H_control[k]
struct ControlTerm {
  ChannelId channel;
  DenseOp H;
};

// A Lindblad collapse operator C_n.
// MCWF uses these via C_n and (C_n^† C_n).
struct CollapseOp {
  DenseOp C;
};

// ModelBase is the minimal "effective model" container for Phase 1 solvers.
//
// It stores already-embedded full-system operators (DenseOp) for:
// - drift Hamiltonian terms (time-independent)
// - control Hamiltonian terms, keyed by ChannelId
// - collapse operators (for Lindblad / MCWF)
//
// Notes:
// - This is intentionally a simple struct (C-like C++).
// - Effective model builders (generic_qubit, oscillator, transmon, etc.) can either:
//   (a) directly populate a ModelBase instance, or
//   (b) wrap it as a field and expose `as_model_base()`.
//
// Phase 1 convention (recommended):
// - At most one ControlTerm per ChannelId (validated by validate()).
//   Phase 2 may relax this on the PulseProgram side (multiple pulses per channel),
//   but typically the model still has one operator per channel.
struct ModelBase {
  // Total Hilbert space dimension of the state vector (size of |psi>).
  Dim hilbert_dim = 0;

  // Optional: subsystem dimensions (e.g., {2,2} for 2 qubits, {3,3,3} for 3 qutrits).
  // If provided, product(subsystem_dims) must equal hilbert_dim.
  std::vector<Dim> subsystem_dims;

  // Drift Hamiltonian terms (summed).
  std::vector<DenseOp> drift_terms;

  // Control Hamiltonian terms (each multiplied by its channel coefficient u(t)).
  std::vector<ControlTerm> control_terms;

  // Collapse operators (noise).
  std::vector<CollapseOp> collapse_ops;

  // Views (Span-based)
  Span<const Dim> subsystem_dims_view() const {
    if (subsystem_dims.empty()) return Span<const Dim>(nullptr, 0);
    return Span<const Dim>(subsystem_dims.data(), static_cast<Dim>(subsystem_dims.size()));
  }

  Span<const DenseOp> drift_terms_view() const {
    if (drift_terms.empty()) return Span<const DenseOp>(nullptr, 0);
    return Span<const DenseOp>(drift_terms.data(), static_cast<Dim>(drift_terms.size()));
  }

  Span<const ControlTerm> control_terms_view() const {
    if (control_terms.empty()) return Span<const ControlTerm>(nullptr, 0);
    return Span<const ControlTerm>(control_terms.data(), static_cast<Dim>(control_terms.size()));
  }

  Span<const CollapseOp> collapse_ops_view() const {
    if (collapse_ops.empty()) return Span<const CollapseOp>(nullptr, 0);
    return Span<const CollapseOp>(collapse_ops.data(), static_cast<Dim>(collapse_ops.size()));
  }

  // Find a control term by channel (linear scan; small N in Phase 1).
  // Returns nullptr if not found.
  const ControlTerm* find_control(ChannelId ch) const {
    for (const ControlTerm& ct : control_terms) {
      if (ct.channel == ch) return &ct;
    }
    return nullptr;
  }

  // Validate basic invariants (no exceptions).
  // - hilbert_dim > 0
  // - if subsystem_dims provided: product == hilbert_dim
  // - each operator has matching dimension
  // - channel IDs are valid and unique (Phase 1 convention)
  Status validate() const {
    if (hilbert_dim <= 0) {
      return Error(ErrorCode::kInvalidArgument, "ModelBase: hilbert_dim must be > 0");
    }

    if (!subsystem_dims.empty()) {
      uint64_t prod = 1;
      for (Dim d : subsystem_dims) {
        if (d <= 0) {
          return Error(ErrorCode::kInvalidArgument, "ModelBase: subsystem_dims entries must be > 0");
        }
        // Avoid overflow surprises: if prod already exceeds hilbert_dim, it can't match.
        if (prod > static_cast<uint64_t>(hilbert_dim) / static_cast<uint64_t>(d)) {
          return Error(ErrorCode::kInvalidArgument,
                       "ModelBase: product(subsystem_dims) exceeds hilbert_dim");
        }
        prod *= static_cast<uint64_t>(d);
      }
      if (prod != static_cast<uint64_t>(hilbert_dim)) {
        return Error(ErrorCode::kInvalidArgument,
                     "ModelBase: product(subsystem_dims) must equal hilbert_dim");
      }
    }

    for (const DenseOp& H : drift_terms) {
      if (H.dim() != hilbert_dim) {
        return Error(ErrorCode::kInvalidArgument, "ModelBase: drift term dim mismatch");
      }
    }

    // Validate control terms: valid, unique channels, dim matches.
    for (size_t i = 0; i < control_terms.size(); ++i) {
      const ControlTerm& ct = control_terms[i];
      if (!ct.channel.valid()) {
        return Error(ErrorCode::kInvalidArgument, "ModelBase: invalid ChannelId in control_terms");
      }
      if (ct.H.dim() != hilbert_dim) {
        return Error(ErrorCode::kInvalidArgument, "ModelBase: control term dim mismatch");
      }
      for (size_t j = i + 1; j < control_terms.size(); ++j) {
        if (control_terms[j].channel == ct.channel) {
          return Error(ErrorCode::kInvalidArgument,
                       "ModelBase: duplicate ChannelId in control_terms");
        }
      }
    }

    for (const CollapseOp& co : collapse_ops) {
      if (co.C.dim() != hilbert_dim) {
        return Error(ErrorCode::kInvalidArgument, "ModelBase: collapse op dim mismatch");
      }
    }

    return Ok();
  }
};

}  // namespace fluxpulse
