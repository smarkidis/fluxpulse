// tests/unit/test_generic_qubit_model.cpp
#include "fluxpulse/model/effective/generic_qubit.hpp"

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

  {
    ModelBase invalid;
    invalid.hilbert_dim = -1;
    assert(!invalid.validate().ok());
    invalid.hilbert_dim = 2;
    invalid.subsystem_dims = {-2};
    assert(!invalid.validate().ok());
    // Multiplying these dimensions would wrap uint64_t back to 4.
    invalid.hilbert_dim = 4;
    invalid.subsystem_dims = {4, (fluxpulse::Dim{1} << 62) + 1};
    assert(!invalid.validate().ok());
  }

  // ---- Default controls: DriveX only; drift is 0.5*(omega+detuning)*Z ----
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{10.0};
    p.detuning = Real{2.0};
    p.t1 = Real{0};
    p.tphi = Real{0};
    p.enable_drive_x = true;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &m);
    assert(st.ok());
    assert(m.validate().ok());

    assert(m.hilbert_dim == 2);
    assert(m.subsystem_dims.size() == 1);
    assert(m.subsystem_dims[0] == 2);

    // Drift term
    assert(m.drift_terms.size() == 1);
    const DenseOp& H0 = m.drift_terms[0];
    assert(H0.dim() == 2);

    const Real w = p.omega + p.detuning;
    const Real h = Real{0.5} * w;

    // H0 = h * Z = diag(h, -h)
    expect_complex_near(at(H0, 0, 0), h, Real{0}, kEps);
    expect_complex_near(at(H0, 1, 1), -h, Real{0}, kEps);
    expect_complex_near(at(H0, 0, 1), Real{0}, Real{0}, kEps);
    expect_complex_near(at(H0, 1, 0), Real{0}, Real{0}, kEps);

    // Control terms
    assert(m.control_terms.size() == 1);
    const auto* cx = m.find_control(ChannelId::DriveX());
    assert(cx != nullptr);
    assert(m.find_control(ChannelId::DriveY()) == nullptr);
    assert(m.find_control(ChannelId::DriveZ()) == nullptr);

    // Hx = 0.5 * X = [[0,0.5],[0.5,0]]
    const DenseOp& Hx = cx->H;
    expect_complex_near(at(Hx, 0, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Hx, 1, 1), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Hx, 0, 1), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(Hx, 1, 0), Real{0.5}, Real{0}, kEps);

    // No noise
    assert(m.collapse_ops.empty());
  }

  // ---- Enable Y and Z controls too ----
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{1.0};
    p.detuning = Real{0.0};
    p.enable_drive_x = true;
    p.enable_drive_y = true;
    p.enable_drive_z = true;

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &m);
    assert(st.ok());

    assert(m.control_terms.size() == 3);
    const auto* cx = m.find_control(ChannelId::DriveX());
    const auto* cy = m.find_control(ChannelId::DriveY());
    const auto* cz = m.find_control(ChannelId::DriveZ());
    assert(cx && cy && cz);

    // Hy = 0.5 * Y = [[0,-0.5 i],[0.5 i, 0]]
    const DenseOp& Hy = cy->H;
    expect_complex_near(at(Hy, 0, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Hy, 1, 1), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Hy, 0, 1), Real{0}, Real{-0.5}, kEps);
    expect_complex_near(at(Hy, 1, 0), Real{0}, Real{+0.5}, kEps);

    // Hz = 0.5 * Z = diag(0.5, -0.5)
    const DenseOp& Hz = cz->H;
    expect_complex_near(at(Hz, 0, 0), Real{0.5}, Real{0}, kEps);
    expect_complex_near(at(Hz, 1, 1), Real{-0.5}, Real{0}, kEps);
    expect_complex_near(at(Hz, 0, 1), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Hz, 1, 0), Real{0}, Real{0}, kEps);
  }

  // ---- Noise: T1 + Tphi collapse operators included ----
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{3.0};
    p.detuning = Real{0.0};
    p.enable_drive_x = true;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    // Choose simple times so rates are nice:
    // gamma1 = 1/t1 = 4  -> sqrt = 2
    // gamma_phi = 1/tphi = 8 -> sqrt(gamma_phi/2) = sqrt(4) = 2
    p.t1 = Real{0.25};
    p.tphi = Real{0.125};

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &m);
    assert(st.ok());

    assert(m.collapse_ops.size() == 2);

    // We don't enforce ordering, but in our builder we push T1 then Tphi.
    const DenseOp& C1 = m.collapse_ops[0].C;
    const DenseOp& Cphi = m.collapse_ops[1].C;

    assert(C1.dim() == 2);
    assert(Cphi.dim() == 2);

    // C1 = sqrt(gamma1) * |0><1| => element (0,1) = 2
    expect_complex_near(at(C1, 0, 1), Real{2.0}, Real{0}, kEps);
    expect_complex_near(at(C1, 0, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(C1, 1, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(C1, 1, 1), Real{0}, Real{0}, kEps);

    // Cphi = sqrt(gamma_phi/2) * Z => diag(+2, -2)
    expect_complex_near(at(Cphi, 0, 0), Real{2.0}, Real{0}, kEps);
    expect_complex_near(at(Cphi, 1, 1), Real{-2.0}, Real{0}, kEps);
    expect_complex_near(at(Cphi, 0, 1), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Cphi, 1, 0), Real{0}, Real{0}, kEps);
  }

  // ---- Disable DriveX: no control terms ----
  {
    fluxpulse::effective::GenericQubitParams p;
    p.omega = Real{1.0};
    p.enable_drive_x = false;
    p.enable_drive_y = false;
    p.enable_drive_z = false;

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_generic_qubit_model(p, &m);
    assert(st.ok());
    assert(m.control_terms.empty());
  }

  return 0;
}
