// tests/unit/test_duffing_transmon_model.cpp
#include "fluxpulse/model/effective/duffing_transmon.hpp"

#include <cassert>
#include <cmath>

#include "fluxpulse/math/complex.hpp"

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline void expect_complex_near(const fluxpulse::Complex& z,
                                fluxpulse::Real re,
                                fluxpulse::Real im,
                                fluxpulse::Real eps) {
  expect_near(z.re, re, eps);
  expect_near(z.im, im, eps);
}

inline const fluxpulse::Complex& at(const fluxpulse::DenseOp& A, int row, int col) {
  return A.data()[row + col * A.lda()];
}

inline fluxpulse::Real rsqrt_int(int x) {
  return static_cast<fluxpulse::Real>(std::sqrt(static_cast<double>(x)));
}

}  // namespace

int main() {
  using fluxpulse::ChannelId;
  using fluxpulse::DenseOp;
  using fluxpulse::ModelBase;
  using fluxpulse::Real;

  constexpr Real kEps = Real{1e-12};

  // ---- Invalid: levels < 2 should fail ----
  {
    fluxpulse::effective::DuffingTransmonParams p;
    p.levels = 1;
    p.omega = Real{1};
    p.anharm = Real{-0.1};

    ModelBase m;
    const fluxpulse::Status st = fluxpulse::effective::build_duffing_transmon_model(p, &m);
    assert(!st.ok());
  }

  // ---- Build: drift diagonal correctness ----
  {
    fluxpulse::effective::DuffingTransmonParams p;
    p.levels = 4;
    p.omega = Real{5.0};
    p.anharm = Real{-0.5};
    p.drive_scale = Real{2.0};
    p.enable_drive_x = true;
    p.enable_drive_y = true;
    p.t1 = Real{0};
    p.tphi = Real{0};

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_duffing_transmon_model(p, &m);
    assert(st.ok());
    assert(m.validate().ok());

    assert(m.hilbert_dim == 4);
    assert(m.subsystem_dims.size() == 1);
    assert(m.subsystem_dims[0] == 4);

    assert(m.drift_terms.size() == 1);
    const DenseOp& H0 = m.drift_terms[0];
    assert(H0.dim() == 4);

    // H0 diag: E_m = omega*m + 0.5*anharm*m*(m-1)
    for (int mlevel = 0; mlevel < 4; ++mlevel) {
      const Real mreal = static_cast<Real>(mlevel);
      const Real expected =
          p.omega * mreal + Real{0.5} * p.anharm * mreal * (mreal - Real{1});
      expect_complex_near(at(H0, mlevel, mlevel), expected, Real{0}, kEps);
    }

    // off-diagonals should be 0 (drift is diagonal)
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        if (r == c) continue;
        expect_complex_near(at(H0, r, c), Real{0}, Real{0}, kEps);
      }
    }
  }

  // ---- Drive operators: Hx = s*(a + a†), Hy = s*i(a† - a) ----
  {
    fluxpulse::effective::DuffingTransmonParams p;
    p.levels = 4;
    p.omega = Real{1.0};
    p.anharm = Real{-0.2};
    p.drive_scale = Real{3.0};
    p.enable_drive_x = true;
    p.enable_drive_y = true;
    p.t1 = Real{0};
    p.tphi = Real{0};

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_duffing_transmon_model(p, &m);
    assert(st.ok());

    const auto* cx = m.find_control(ChannelId::DriveX());
    const auto* cy = m.find_control(ChannelId::DriveY());
    assert(cx != nullptr);
    assert(cy != nullptr);

    const DenseOp& Hx = cx->H;
    const DenseOp& Hy = cy->H;
    assert(Hx.dim() == 4);
    assert(Hy.dim() == 4);

    // Expected:
    // a has (m, m+1) = sqrt(m+1)
    // a† has (m+1, m) = sqrt(m+1)
    //
    // Hx superdiag (m,m+1) = s*sqrt(m+1)
    // Hx subdiag   (m+1,m) = s*sqrt(m+1)
    //
    // Hy superdiag (m,m+1) = s*(-i)*sqrt(m+1)
    // Hy subdiag   (m+1,m) = s*(+i)*sqrt(m+1)
    const Real s = p.drive_scale;

    // Diagonal should be 0 for both Hx,Hy.
    for (int d = 0; d < 4; ++d) {
      expect_complex_near(at(Hx, d, d), Real{0}, Real{0}, kEps);
      expect_complex_near(at(Hy, d, d), Real{0}, Real{0}, kEps);
    }

    // Check nearest off-diagonals.
    for (int mlevel = 0; mlevel + 1 < 4; ++mlevel) {
      const Real a01 = rsqrt_int(mlevel + 1);
      const Real v = s * a01;

      // Hx
      expect_complex_near(at(Hx, mlevel, mlevel + 1), v, Real{0}, kEps);
      expect_complex_near(at(Hx, mlevel + 1, mlevel), v, Real{0}, kEps);

      // Hy
      expect_complex_near(at(Hy, mlevel, mlevel + 1), Real{0}, -v, kEps);  // -i*v
      expect_complex_near(at(Hy, mlevel + 1, mlevel), Real{0}, +v, kEps);  // +i*v
    }

    // All other entries should be 0.
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        const bool is_diag = (r == c);
        const bool is_nn = (std::abs(r - c) == 1);
        if (is_diag || is_nn) continue;

        expect_complex_near(at(Hx, r, c), Real{0}, Real{0}, kEps);
        expect_complex_near(at(Hy, r, c), Real{0}, Real{0}, kEps);
      }
    }
  }

  // ---- Noise: relaxation C = sqrt(gamma1)*a, dephasing C = sqrt(gamma_phi/2)*n ----
  {
    fluxpulse::effective::DuffingTransmonParams p;
    p.levels = 4;
    p.omega = Real{1.0};
    p.anharm = Real{-0.2};
    p.drive_scale = Real{1.0};
    p.enable_drive_x = true;
    p.enable_drive_y = false;

    // Choose times so rates are nice:
    // gamma1 = 1/t1 = 4  => sqrt = 2
    // gamma_phi = 1/tphi = 8 => sqrt(gamma_phi/2) = sqrt(4) = 2
    p.t1 = Real{0.25};
    p.tphi = Real{0.125};

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_duffing_transmon_model(p, &m);
    assert(st.ok());

    assert(m.collapse_ops.size() == 2);

    const DenseOp& C1 = m.collapse_ops[0].C;    // relaxation
    const DenseOp& Cphi = m.collapse_ops[1].C;  // dephasing
    assert(C1.dim() == 4);
    assert(Cphi.dim() == 4);

    // C1 = 2*a so (m,m+1) = 2*sqrt(m+1)
    for (int mlevel = 0; mlevel + 1 < 4; ++mlevel) {
      const Real expected = Real{2.0} * rsqrt_int(mlevel + 1);
      expect_complex_near(at(C1, mlevel, mlevel + 1), expected, Real{0}, kEps);
    }
    // Other entries should be 0 (including diagonal and subdiag).
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        if (c == r + 1) continue;  // superdiag checked above
        expect_complex_near(at(C1, r, c), Real{0}, Real{0}, kEps);
      }
    }

    // Cphi = 2*n => diag(0,2,4,6)
    expect_complex_near(at(Cphi, 0, 0), Real{0}, Real{0}, kEps);
    expect_complex_near(at(Cphi, 1, 1), Real{2}, Real{0}, kEps);
    expect_complex_near(at(Cphi, 2, 2), Real{4}, Real{0}, kEps);
    expect_complex_near(at(Cphi, 3, 3), Real{6}, Real{0}, kEps);

    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        if (r == c) continue;
        expect_complex_near(at(Cphi, r, c), Real{0}, Real{0}, kEps);
      }
    }
  }

  // ---- Control flags: disable DriveY ----
  {
    fluxpulse::effective::DuffingTransmonParams p;
    p.levels = 3;
    p.omega = Real{1.0};
    p.anharm = Real{-0.3};
    p.enable_drive_x = true;
    p.enable_drive_y = false;

    ModelBase m;
    fluxpulse::Status st = fluxpulse::effective::build_duffing_transmon_model(p, &m);
    assert(st.ok());

    assert(m.find_control(ChannelId::DriveX()) != nullptr);
    assert(m.find_control(ChannelId::DriveY()) == nullptr);
  }

  return 0;
}