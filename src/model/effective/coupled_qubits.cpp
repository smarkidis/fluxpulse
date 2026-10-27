// src/model/effective/coupled_qubits.cpp
#include "fluxpulse/model/effective/coupled_qubits.hpp"

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/model/noise/collapse_ops.hpp"
#include "fluxpulse/operators/tensor.hpp"

namespace fluxpulse {
namespace effective {
namespace {

inline Complex cplx(Real re, Real im) {
  Complex z;
  z.re = re;
  z.im = im;
  return z;
}

inline void denseop_set_zero(DenseOp* A) {
  assert(A != nullptr);
  const Dim n = A->dim();
  const Dim lda = A->lda();
  Complex* data = A->data();
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      data[row + col * lda] = cplx(Real{0}, Real{0});
    }
  }
}

inline void denseop_scale_inplace(DenseOp* A, Real s) {
  assert(A != nullptr);
  const Dim n = A->dim();
  const Dim lda = A->lda();
  Complex* data = A->data();
  for (Dim col = 0; col < n; ++col) {
    for (Dim row = 0; row < n; ++row) {
      Complex& z = data[row + col * lda];
      z.re *= s;
      z.im *= s;
    }
  }
}

DenseOp pauli_x() {
  // X = [[0,1],[1,0]]
  DenseOp A(2);
  denseop_set_zero(&A);
  A.data()[0 + 1 * A.lda()] = cplx(Real{1}, Real{0});
  A.data()[1 + 0 * A.lda()] = cplx(Real{1}, Real{0});
  return A;
}

DenseOp pauli_y() {
  // Y = [[0,-i],[i,0]]
  DenseOp A(2);
  denseop_set_zero(&A);
  A.data()[0 + 1 * A.lda()] = cplx(Real{0}, Real{-1});  // -i
  A.data()[1 + 0 * A.lda()] = cplx(Real{0}, Real{+1});  // +i
  return A;
}

DenseOp pauli_z() {
  // Z = [[1,0],[0,-1]]
  DenseOp A(2);
  denseop_set_zero(&A);
  A.data()[0 + 0 * A.lda()] = cplx(Real{1}, Real{0});
  A.data()[1 + 1 * A.lda()] = cplx(Real{-1}, Real{0});
  return A;
}

inline bool vec_empty_or_size(const std::vector<Real>& v, Index n) {
  return v.empty() || static_cast<Index>(v.size()) == n;
}
inline bool ch_empty_or_size(const std::vector<ChannelId>& v, Index n) {
  return v.empty() || static_cast<Index>(v.size()) == n;
}

inline Dim pow2_dim(Index nqubits) {
  Dim d = 1;
  for (Index i = 0; i < nqubits; ++i) d = static_cast<Dim>(d * 2);
  return d;
}

inline Status validate_pair(Index nqubits, const QubitPairCoupling& c, const char* where) {
  (void)where;
  if (c.q0 < 0 || c.q1 < 0) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: negative qubit index");
  }
  if (c.q0 == c.q1) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: coupling requires q0 != q1");
  }
  if (c.q0 >= nqubits || c.q1 >= nqubits) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: coupling index out of range");
  }
  return Ok();
}

// Default multi-qubit channel mapping:
//   axis_base in {1,2,3} for {X,Y,Z}, then add 3*qubit.
// Ensures qubit 0 matches GenericQubit: 1/2/3.
inline Status default_drive_channel(uint16_t axis_base, Index qubit, ChannelId* out) {
  if (!out) {
    return Error(ErrorCode::kInvalidArgument, "default_drive_channel: out is null");
  }
  if (qubit < 0) {
    return Error(ErrorCode::kInvalidArgument, "default_drive_channel: negative qubit");
  }
  // Compute in 32-bit to check overflow into uint16_t.
  const uint32_t q = static_cast<uint32_t>(qubit);
  const uint32_t id = static_cast<uint32_t>(axis_base) + 3u * q;
  if (id == 0u || id > 0xFFFFu) {
    return Error(ErrorCode::kInvalidArgument, "default_drive_channel: ChannelId overflow");
  }
  *out = ChannelId{static_cast<uint16_t>(id)};
  return Ok();
}

// Add per-qubit controls for an axis.
// - If explicit_channels is empty: generate default channels for all qubits, but only if enabled.
// - If explicit_channels provided: must be size n, and only valid entries generate terms.
Status add_axis_controls(bool enabled,
                         const std::vector<ChannelId>& explicit_channels,
                         uint16_t axis_base,
                         const DenseOp& pauli,
                         Index nqubits,
                         std::vector<ControlTerm>* out_terms) {
  if (!enabled && explicit_channels.empty()) return Ok();
  if (!out_terms) {
    return Error(ErrorCode::kInvalidArgument, "add_axis_controls: out_terms is null");
  }

  if (!explicit_channels.empty() && static_cast<Index>(explicit_channels.size()) != nqubits) {
    return Error(ErrorCode::kInvalidArgument, "add_axis_controls: explicit channel vector size mismatch");
  }

  for (Index i = 0; i < nqubits; ++i) {
    ChannelId ch = ChannelId::Invalid();
    if (!explicit_channels.empty()) {
      ch = explicit_channels[static_cast<std::size_t>(i)];
      if (!ch.valid()) continue;  // explicit invalid => skip this qubit
    } else {
      if (!enabled) continue;
      Status st = default_drive_channel(axis_base, i, &ch);
      if (!st.ok()) return Annotate(st, "add_axis_controls: default_drive_channel");
    }

    DenseOp Hloc = pauli;
    denseop_scale_inplace(&Hloc, Real{0.5});
    DenseOp Hfull = embed_qubit_op(Hloc, i, nqubits);

    ControlTerm ct;
    ct.channel = ch;
    ct.H = std::move(Hfull);
    out_terms->push_back(std::move(ct));
  }

  return Ok();
}

}  // namespace

Status build_coupled_qubits_model(const CoupledQubitsParams& p, ModelBase* out) {
  if (!out) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: out is null");
  }
  if (p.nqubits < 1) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: nqubits must be >= 1");
  }

  const Index n = p.nqubits;

  if (!vec_empty_or_size(p.omega, n)) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: omega must be empty or size == nqubits");
  }
  if (!vec_empty_or_size(p.detuning, n)) {
    return Error(ErrorCode::kInvalidArgument,
                 "build_coupled_qubits_model: detuning must be empty or size == nqubits");
  }
  if (!vec_empty_or_size(p.t1, n) || !vec_empty_or_size(p.tphi, n)) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: t1/tphi must be empty or size == nqubits");
  }
  if (!ch_empty_or_size(p.drive_x, n) || !ch_empty_or_size(p.drive_y, n) || !ch_empty_or_size(p.drive_z, n)) {
    return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: drive_* must be empty or size == nqubits");
  }

  if (n < 2) {
    if (!p.exchange_xy.empty() || !p.zz.empty()) {
      return Error(ErrorCode::kInvalidArgument, "build_coupled_qubits_model: couplings require nqubits >= 2");
    }
  }

  for (const QubitPairCoupling& c : p.exchange_xy) {
    Status st = validate_pair(n, c, "exchange_xy");
    if (!st.ok()) return st;
  }
  for (const QubitPairCoupling& c : p.zz) {
    Status st = validate_pair(n, c, "zz");
    if (!st.ok()) return st;
  }

  ModelBase m;
  m.hilbert_dim = pow2_dim(n);
  m.subsystem_dims.assign(static_cast<std::size_t>(n), Dim{2});
  m.drift_terms.clear();
  m.control_terms.clear();
  m.collapse_ops.clear();

  const DenseOp X = pauli_x();
  const DenseOp Y = pauli_y();
  const DenseOp Z = pauli_z();

  // ---------------------------------------------------------------------------
  // Drift: sum_i 0.5*(omega[i]+detuning[i]) * Z_i
  // (We always add the term per qubit, consistent with GenericQubit always adding one drift term.)
  // ---------------------------------------------------------------------------
  for (Index i = 0; i < n; ++i) {
    const Real w = (p.omega.empty() ? Real{0} : p.omega[static_cast<std::size_t>(i)]) +
                   (p.detuning.empty() ? Real{0} : p.detuning[static_cast<std::size_t>(i)]);
    DenseOp Hloc = Z;
    denseop_scale_inplace(&Hloc, Real{0.5} * w);
    DenseOp Hfull = embed_qubit_op(Hloc, i, n);
    m.drift_terms.push_back(std::move(Hfull));
  }

  // ---------------------------------------------------------------------------
  // Controls: per-qubit 0.5*X / 0.5*Y / 0.5*Z, each keyed by a unique ChannelId.
  // ---------------------------------------------------------------------------
  {
    Status st = add_axis_controls(p.enable_drive_x, p.drive_x, /*axis_base=*/1, X, n, &m.control_terms);
    if (!st.ok()) return Annotate(st, "build_coupled_qubits_model: add_axis_controls(X)");
  }
  {
    Status st = add_axis_controls(p.enable_drive_y, p.drive_y, /*axis_base=*/2, Y, n, &m.control_terms);
    if (!st.ok()) return Annotate(st, "build_coupled_qubits_model: add_axis_controls(Y)");
  }
  {
    Status st = add_axis_controls(p.enable_drive_z, p.drive_z, /*axis_base=*/3, Z, n, &m.control_terms);
    if (!st.ok()) return Annotate(st, "build_coupled_qubits_model: add_axis_controls(Z)");
  }

  // ---------------------------------------------------------------------------
  // Couplings (drift terms)
  //   exchange_xy: strength*(X_i X_j + Y_i Y_j)
  //   zz:          strength*(Z_i Z_j)
  // Uses embed_two_qubit_product from operators/tensor.hpp.
  // ---------------------------------------------------------------------------
  for (const QubitPairCoupling& c : p.exchange_xy) {
    if (c.strength == Real{0}) continue;
    DenseOp XX = embed_two_qubit_product(X, X, c.q0, c.q1, n);
    DenseOp YY = embed_two_qubit_product(Y, Y, c.q0, c.q1, n);
    denseop_scale_inplace(&XX, c.strength);
    denseop_scale_inplace(&YY, c.strength);
    m.drift_terms.push_back(std::move(XX));
    m.drift_terms.push_back(std::move(YY));
  }

  for (const QubitPairCoupling& c : p.zz) {
    if (c.strength == Real{0}) continue;
    DenseOp ZZ = embed_two_qubit_product(Z, Z, c.q0, c.q1, n);
    denseop_scale_inplace(&ZZ, c.strength);
    m.drift_terms.push_back(std::move(ZZ));
  }

  // ---------------------------------------------------------------------------
  // Noise: embed single-qubit collapse ops into full space.
  // ---------------------------------------------------------------------------
  for (Index i = 0; i < n; ++i) {
    const Real t1 = p.t1.empty() ? Real{0} : p.t1[static_cast<std::size_t>(i)];
    const Real tphi = p.tphi.empty() ? Real{0} : p.tphi[static_cast<std::size_t>(i)];

    if (t1 > Real{0}) {
      const Real gamma1 = Real{1} / t1;
      DenseOp C2;
      Status st = noise::qubit_amplitude_damping(gamma1, &C2);
      if (!st.ok()) return Annotate(st, "build_coupled_qubits_model: amplitude damping");

      DenseOp Cfull = embed_qubit_op(C2, i, n);
      CollapseOp co;
      co.C = std::move(Cfull);
      m.collapse_ops.push_back(std::move(co));
    }

    if (tphi > Real{0}) {
      const Real gamma_phi = Real{1} / tphi;
      DenseOp C2;
      Status st = noise::qubit_pure_dephasing(gamma_phi, &C2);
      if (!st.ok()) return Annotate(st, "build_coupled_qubits_model: pure dephasing");

      DenseOp Cfull = embed_qubit_op(C2, i, n);
      CollapseOp co;
      co.C = std::move(Cfull);
      m.collapse_ops.push_back(std::move(co));
    }
  }

  // Final validation (dims + channel uniqueness)
  Status st = m.validate();
  if (!st.ok()) return Annotate(st, "build_coupled_qubits_model: model.validate");

  *out = std::move(m);
  return Ok();
}

}  // namespace effective
}  // namespace fluxpulse