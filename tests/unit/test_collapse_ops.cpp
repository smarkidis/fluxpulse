// tests/unit/test_collapse_ops.cpp
#include "fluxpulse/model/noise/collapse_ops.hpp"

#include <cassert>

#include "fluxpulse/math/complex.hpp"

namespace {

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

inline void expect_complex_near(const fluxpulse::Complex& a,
                                const fluxpulse::Complex& b,
                                fluxpulse::Real eps) {
  expect_near(a.re, b.re, eps);
  expect_near(a.im, b.im, eps);
}

inline fluxpulse::Complex conjc(fluxpulse::Complex z) {
  fluxpulse::Complex w;
  w.re = z.re;
  w.im = -z.im;
  return w;
}

inline fluxpulse::Complex cmul(const fluxpulse::Complex& a, const fluxpulse::Complex& b) {
  fluxpulse::Complex r;
  r.re = a.re * b.re - a.im * b.im;
  r.im = a.re * b.im + a.im * b.re;
  return r;
}

inline fluxpulse::Complex cadd(const fluxpulse::Complex& a, const fluxpulse::Complex& b) {
  fluxpulse::Complex r;
  r.re = a.re + b.re;
  r.im = a.im + b.im;
  return r;
}

}  // namespace

int main() {
  using fluxpulse::DenseOp;
  using fluxpulse::Real;

  constexpr Real kEps = Real{1e-12};

  // ---- sigma_minus structure ----
  {
    DenseOp sm = fluxpulse::noise::qubit_sigma_minus();
    assert(sm.dim() == 2);

    const auto idx = [&](int row, int col) -> fluxpulse::Complex {
      return sm.data()[row + col * sm.lda()];
    };

    // Expect |0><1|:
    // [[0,1],
    //  [0,0]]
    expect_complex_near(idx(0, 0), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(idx(0, 1), fluxpulse::Complex{Real{1}, Real{0}}, kEps);
    expect_complex_near(idx(1, 0), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(idx(1, 1), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
  }

  // ---- sigma_z structure ----
  {
    DenseOp sz = fluxpulse::noise::qubit_sigma_z();
    assert(sz.dim() == 2);

    const auto idx = [&](int row, int col) -> fluxpulse::Complex {
      return sz.data()[row + col * sz.lda()];
    };

    // diag(1, -1)
    expect_complex_near(idx(0, 0), fluxpulse::Complex{Real{1}, Real{0}}, kEps);
    expect_complex_near(idx(0, 1), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(idx(1, 0), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(idx(1, 1), fluxpulse::Complex{Real{-1}, Real{0}}, kEps);
  }

  // ---- Amplitude damping: C = sqrt(gamma1) * sigma_minus ----
  {
    const Real gamma1 = Real{4.0};  // sqrt = 2.0
    DenseOp C;
    const fluxpulse::Status st = fluxpulse::noise::qubit_amplitude_damping(gamma1, &C);
    assert(st.ok());
    assert(C.dim() == 2);

    const auto idx = [&](int row, int col) -> fluxpulse::Complex {
      return C.data()[row + col * C.lda()];
    };

    // Expect element (0,1) = 2.0
    expect_complex_near(idx(0, 1), fluxpulse::Complex{Real{2.0}, Real{0}}, kEps);
    expect_complex_near(idx(0, 0), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(idx(1, 0), fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(idx(1, 1), fluxpulse::Complex{Real{0}, Real{0}}, kEps);

    // Check C†C = diag(0, gamma1)
    // compute (C†C)_{ij} = sum_k conj(C_{k,i}) * C_{k,j}
    fluxpulse::Complex cdc00{Real{0}, Real{0}};
    fluxpulse::Complex cdc01{Real{0}, Real{0}};
    fluxpulse::Complex cdc10{Real{0}, Real{0}};
    fluxpulse::Complex cdc11{Real{0}, Real{0}};
    for (int k = 0; k < 2; ++k) {
      const fluxpulse::Complex cki0 = idx(k, 0);
      const fluxpulse::Complex cki1 = idx(k, 1);
      const fluxpulse::Complex ck0 = cki0;
      const fluxpulse::Complex ck1 = cki1;

      // i=0,j=0
      cdc00 = cadd(cdc00, cmul(conjc(ck0), ck0));
      // i=0,j=1
      cdc01 = cadd(cdc01, cmul(conjc(ck0), ck1));
      // i=1,j=0
      cdc10 = cadd(cdc10, cmul(conjc(ck1), ck0));
      // i=1,j=1
      cdc11 = cadd(cdc11, cmul(conjc(ck1), ck1));
    }

    expect_complex_near(cdc00, fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(cdc01, fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    expect_complex_near(cdc10, fluxpulse::Complex{Real{0}, Real{0}}, kEps);
    // Check C†C (1,1) = gamma1
    expect_near(cdc11.re, gamma1, kEps);
    expect_near(cdc11.im, Real{0}, kEps);
    //expect_complex_near(cdc11, fluxpulse::Complex{gamma1}, fluxpulse::Complex{Real{0}, Real{0}}.im, kEps); // (re=gamma1, im=0)
    // The above line is awkward due to aggregate init; do explicit:
    expect_near(cdc11.re, gamma1, kEps);
    expect_near(cdc11.im, Real{0}, kEps);
  }

  // ---- Pure dephasing: C = sqrt(gamma_phi/2) * sigma_z ----
  {
    const Real gamma_phi = Real{8.0};
    // scale = sqrt(4) = 2
    DenseOp C;
    const fluxpulse::Status st = fluxpulse::noise::qubit_pure_dephasing(gamma_phi, &C);
    assert(st.ok());
    assert(C.dim() == 2);

    const auto idx = [&](int row, int col) -> fluxpulse::Complex {
      return C.data()[row + col * C.lda()];
    };

    // Expect diag(+2, -2)
    expect_near(idx(0, 0).re, Real{2.0}, kEps);
    expect_near(idx(0, 0).im, Real{0.0}, kEps);
    expect_near(idx(1, 1).re, Real{-2.0}, kEps);
    expect_near(idx(1, 1).im, Real{0.0}, kEps);

    // Off-diagonals zero
    expect_near(idx(0, 1).re, Real{0.0}, kEps);
    expect_near(idx(0, 1).im, Real{0.0}, kEps);
    expect_near(idx(1, 0).re, Real{0.0}, kEps);
    expect_near(idx(1, 0).im, Real{0.0}, kEps);

    // C†C should be (gamma_phi/2) * I (since sigma_z^2 = I and C is real diagonal)
    const Real expected = gamma_phi * Real{0.5};

    // Compute diagonal entries of C†C quickly:
    // (C†C)_{jj} = sum_k conj(C_{k,j}) * C_{k,j}
    // Since C is diagonal, this is |C_{j,j}|^2.
    const Real d0 = idx(0, 0).re * idx(0, 0).re + idx(0, 0).im * idx(0, 0).im;
    const Real d1 = idx(1, 1).re * idx(1, 1).re + idx(1, 1).im * idx(1, 1).im;

    expect_near(d0, expected, kEps);
    expect_near(d1, expected, kEps);
  }

  // ---- Invalid gamma values should fail ----
  {
    DenseOp C;
    fluxpulse::Status st = fluxpulse::noise::qubit_amplitude_damping(Real{0.0}, &C);
    assert(!st.ok());
    st = fluxpulse::noise::qubit_amplitude_damping(Real{-1.0}, &C);
    assert(!st.ok());

    st = fluxpulse::noise::qubit_pure_dephasing(Real{0.0}, &C);
    assert(!st.ok());
    st = fluxpulse::noise::qubit_pure_dephasing(Real{-1.0}, &C);
    assert(!st.ok());
  }

  return 0;
}