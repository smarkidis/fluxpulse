// tests/unit/test_coupled_qubits_model.cpp
#include "fluxpulse/model/effective/coupled_qubits.hpp"

#include <cassert>

#include "fluxpulse/math/complex.hpp"

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline void expect_complex_near(const fluxpulse::Complex& a,
                                fluxpulse::Real re,
                                fluxpulse::Real im,
                                fluxpulse::Real eps) {
  expect_near(a.re, re, eps);
  expect_near(a.im, im, eps);
}

inline const fluxpulse::Complex& at(const fluxpulse::DenseOp& A, int row, int col) {
  return A.data()[row + col * A.lda()];
}

}  // namespace

int main() {
  using fluxpulse::ChannelId;
  using fluxpulse::DenseOp;
  using fluxpulse::ModelBase;
  using fluxpulse::Real;

  constexpr Real kEps = Real{1e-12};

  // ---------------------------------------------------------------------------
  // Basic 2Q build: drift(Z0,Z1) + controls(X0,X1) + exchange + ZZ, no noise.
  // Validates:
  // - little-endian embedding
  // - default multi-qubit channel mapping (1+3*i)
  // - coupling matrices (XX, YY, ZZ)
  // ---------------------------------------------------------------------------
  {
    fluxpulse::effective::CoupledQubitsParams p;
    p.nqubits = 2;

    // Drift: H0 = 0.5*(omega+detuning)*Z per qubit
    p.omega = {Real{10.0}, Real{20.0}};
    p.detuning = {Real{2.0}, Real{3.0}};

    // Controls: default channels for X only.
    p.enable_drive_x = true;
    p.enable_drive_y = false;
    p.enable_drive_z = false;
    p.drive_x.clear();
    p.drive_y.clear();
    p.drive_z.clear();

    // Couplings
    const Real J = Real{0.7};
    const Real zeta = Real{0.11};
    p.exchange_xy.push_back({/*q0=*/0, /*q1=*/1, /*strength=*/J});
    p.zz.push_back({/*q0=*/0, /*q1=*/1, /*strength=*/zeta});

    // No noise
    p.t1.clear();
    p.tphi.clear();

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &m);
    assert(st.ok());
    assert(m.validate().ok());

    assert(m.hilbert_dim == 4);
    assert(m.subsystem_dims.size() == 2);
    assert(m.subsystem_dims[0] == 2);
    assert(m.subsystem_dims[1] == 2);

    // Drift terms:
    // - one Z term per qubit (2)
    // - exchange adds XX and YY (2)
    // - ZZ adds ZZ (1)
    assert(m.drift_terms.size() == 5);

    // Control terms: one per qubit for DriveX
    assert(m.control_terms.size() == 2);

    // Default channels for 2 qubits:
    // qubit 0: 1 + 3*0 = 1
    // qubit 1: 1 + 3*1 = 4
    const auto* hx0 = m.find_control(ChannelId{1});
    const auto* hx1 = m.find_control(ChannelId{4});
    assert(hx0 != nullptr);
    assert(hx1 != nullptr);
    assert(m.find_control(ChannelId::DriveY()) == nullptr);
    assert(m.find_control(ChannelId::DriveZ()) == nullptr);

    // ----------------------
    // Drift Z0 and Z1 checks
    // Basis order (little-endian): |q1 q0> => [|00>,|01>,|10>,|11>]
    // Z0 diag: [+,-,+,-]
    // Z1 diag: [+,+,-,-]
    // ----------------------
    const Real w0 = p.omega[0] + p.detuning[0];
    const Real w1 = p.omega[1] + p.detuning[1];
    const Real h0 = Real{0.5} * w0;
    const Real h1 = Real{0.5} * w1;

    const DenseOp& HZ0 = m.drift_terms[0];
    const DenseOp& HZ1 = m.drift_terms[1];
    assert(HZ0.dim() == 4);
    assert(HZ1.dim() == 4);

    // Z0
    expect_complex_near(at(HZ0, 0, 0), +h0, Real{0}, kEps);
    expect_complex_near(at(HZ0, 1, 1), -h0, Real{0}, kEps);
    expect_complex_near(at(HZ0, 2, 2), +h0, Real{0}, kEps);
    expect_complex_near(at(HZ0, 3, 3), -h0, Real{0}, kEps);

    // Z1
    expect_complex_near(at(HZ1, 0, 0), +h1, Real{0}, kEps);
    expect_complex_near(at(HZ1, 1, 1), +h1, Real{0}, kEps);
    expect_complex_near(at(HZ1, 2, 2), -h1, Real{0}, kEps);
    expect_complex_near(at(HZ1, 3, 3), -h1, Real{0}, kEps);

    // ----------------------
    // Control X0 and X1 checks
    // Hx0 = 0.5 * X0 couples |00><01| and |10><11|
    // Hx1 = 0.5 * X1 couples |00><10| and |01><11|
    // ----------------------
    const DenseOp& HX0 = hx0->H;
    const DenseOp& HX1 = hx1->H;
    assert(HX0.dim() == 4);
    assert(HX1.dim() == 4);

    // X0:
    expect_complex_near(at(HX0, 0, 1), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(HX0, 1, 0), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(HX0, 2, 3), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(HX0, 3, 2), Real{0.5}, Real{0}, kEps);

    // X1:
    expect_complex_near(at(HX1, 0, 2), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(HX1, 2, 0), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(HX1, 1, 3), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(HX1, 3, 1), Real{0.5}, Real{0}, kEps);

    // ----------------------
    // Coupling checks
    // Exchange was appended as:
    //   Hxx = J * (X0 X1)
    //   Hyy = J * (Y0 Y1)
    // ZZ:
    //   Hzz = zeta * (Z0 Z1)
    // ----------------------
    const DenseOp& HXX = m.drift_terms[2];
    const DenseOp& HYY = m.drift_terms[3];
    const DenseOp& HZZ = m.drift_terms[4];

    // XX has ones on: |00><11|, |01><10|, |10><01|, |11><00|
    expect_complex_near(at(HXX, 0, 3), +J, Real{0}, kEps);
    expect_complex_near(at(HXX, 1, 2), +J, Real{0}, kEps);
    expect_complex_near(at(HXX, 2, 1), +J, Real{0}, kEps);
    expect_complex_near(at(HXX, 3, 0), +J, Real{0}, kEps);

    // YY differs by signs:
    // Y⊗Y maps |00><11| with -1 and |01><10| with +1 (and Hermitian transpose).
    expect_complex_near(at(HYY, 0, 3), -J, Real{0}, kEps);
    expect_complex_near(at(HYY, 1, 2), +J, Real{0}, kEps);
    expect_complex_near(at(HYY, 2, 1), +J, Real{0}, kEps);
    expect_complex_near(at(HYY, 3, 0), -J, Real{0}, kEps);

    // Sanity: the sum (XX + YY) cancels |00><11| and doubles |01><10|
    const fluxpulse::Complex s_03 = fluxpulse::add(at(HXX, 0, 3), at(HYY, 0, 3));
    const fluxpulse::Complex s_12 = fluxpulse::add(at(HXX, 1, 2), at(HYY, 1, 2));
    expect_complex_near(s_03, Real{0}, Real{0}, kEps);
    expect_complex_near(s_12, Real{2} * J, Real{0}, kEps);

    // ZZ diagonal: diag(+,-,-,+) * zeta
    expect_complex_near(at(HZZ, 0, 0), +zeta, Real{0}, kEps);
    expect_complex_near(at(HZZ, 1, 1), -zeta, Real{0}, kEps);
    expect_complex_near(at(HZZ, 2, 2), -zeta, Real{0}, kEps);
    expect_complex_near(at(HZZ, 3, 3), +zeta, Real{0}, kEps);

    // No noise
    assert(m.collapse_ops.empty());
  }

  // ---------------------------------------------------------------------------
  // Noise embedding checks (2 qubits, T1 + Tphi on both qubits)
  // Validates:
  // - per-qubit collapse ops embedded to correct matrix elements/diagonals
  // - ordering: per qubit, push T1 then Tphi (matches builder)
  // ---------------------------------------------------------------------------
  {
    fluxpulse::effective::CoupledQubitsParams p;
    p.nqubits = 2;
    p.omega = {Real{0}, Real{0}};
    p.detuning = {Real{0}, Real{0}};

    // Disable controls for this test
    p.enable_drive_x = false;
    p.enable_drive_y = false;
    p.enable_drive_z = false;
    p.drive_x.clear();
    p.drive_y.clear();
    p.drive_z.clear();

    // No couplings
    p.exchange_xy.clear();
    p.zz.clear();

    // Choose times so amplitudes are integers:
    // qubit0: t1=0.25 => gamma=4 => sqrt=2
    // qubit0: tphi=0.125 => gamma=8 => sqrt(g/2)=2
    // qubit1: t1=1.0 => gamma=1 => sqrt=1
    // qubit1: tphi=0.5 => gamma=2 => sqrt(g/2)=1
    p.t1 = {Real{0.25}, Real{1.0}};
    p.tphi = {Real{0.125}, Real{0.5}};

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &m);
    assert(st.ok());
    assert(m.validate().ok());

    assert(m.hilbert_dim == 4);
    assert(m.collapse_ops.size() == 4);

    const DenseOp& C0_1 = m.collapse_ops[0].C;    // q0 T1
    const DenseOp& C0_p = m.collapse_ops[1].C;    // q0 Tphi
    const DenseOp& C1_1 = m.collapse_ops[2].C;    // q1 T1
    const DenseOp& C1_p = m.collapse_ops[3].C;    // q1 Tphi

    // q0 sigma_- entries: |01>->|00| and |11>->|10|
    // i.e. (row,col) = (0,1) and (2,3)
    expect_complex_near(at(C0_1, 0, 1), Real{2}, Real{0}, kEps);
    expect_complex_near(at(C0_1, 2, 3), Real{2}, Real{0}, kEps);

    // q1 sigma_- entries: |10>->|00| and |11>->|01|
    // i.e. (0,2) and (1,3)
    expect_complex_near(at(C1_1, 0, 2), Real{1}, Real{0}, kEps);
    expect_complex_near(at(C1_1, 1, 3), Real{1}, Real{0}, kEps);

    // q0 Z diag (+,-,+,-) * 2
    expect_complex_near(at(C0_p, 0, 0), Real{+2}, Real{0}, kEps);
    expect_complex_near(at(C0_p, 1, 1), Real{-2}, Real{0}, kEps);
    expect_complex_near(at(C0_p, 2, 2), Real{+2}, Real{0}, kEps);
    expect_complex_near(at(C0_p, 3, 3), Real{-2}, Real{0}, kEps);

    // q1 Z diag (+,+,-,-) * 1
    expect_complex_near(at(C1_p, 0, 0), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(C1_p, 1, 1), Real{+1}, Real{0}, kEps);
    expect_complex_near(at(C1_p, 2, 2), Real{-1}, Real{0}, kEps);
    expect_complex_near(at(C1_p, 3, 3), Real{-1}, Real{0}, kEps);
  }

  // ---------------------------------------------------------------------------
  // Channel uniqueness validation: duplicate ChannelId should fail.
  // ---------------------------------------------------------------------------
  {
    fluxpulse::effective::CoupledQubitsParams p;
    p.nqubits = 2;
    p.omega = {Real{1}, Real{1}};
    p.detuning = {Real{0}, Real{0}};
    p.enable_drive_x = true;

    // Explicit (invalid) duplicate channel IDs:
    p.drive_x = {ChannelId{10}, ChannelId{10}};

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_coupled_qubits_model(p, &m);
    assert(!st.ok());
  }

  return 0;
}