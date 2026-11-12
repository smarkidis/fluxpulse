#include "fluxpulse/linalg/blas_gemv.hpp"

#include <cassert>
#include <limits>

namespace {

using fluxpulse::Complex;
using fluxpulse::Real;

static inline bool near(Real a, Real b, Real tol) noexcept {
  const Real d = (a > b) ? (a - b) : (b - a);
  return d <= tol;
}

static inline bool complex_near(Complex a, Complex b, Real tol) noexcept {
  return near(a.re, b.re, tol) && near(a.im, b.im, tol);
}

} // namespace

int main() {
  using fluxpulse::blas_gemv_colmajor;
  using fluxpulse::Complex;
  using fluxpulse::complex_one;
  using fluxpulse::complex_zero;
  using fluxpulse::Dim;
  using fluxpulse::Real;
  using fluxpulse::Span;

  constexpr Real kEps = Real{1e-12};

  // Non-real scalars, padded column-major matrix, and strided vectors.
  {
    const Complex A[] = {{1, 2},  {3, -1}, {999, 999},
                         {-2, 1}, {4, 3},  {999, 999}};
    const Complex x[] = {{2, -1}, {999, 999}, {-1, 3}};
    Complex y[] = {{1, -2}, {77, 88}, {3, 1}};
    const Complex alpha{0.5, -1}, beta{-0.25, 0.75};
    Complex expected[2];
    for (Dim i = 0; i < 2; ++i) {
      Complex sum{};
      for (Dim j = 0; j < 2; ++j)
        sum = add(sum, mul(A[j * 3 + i], x[j * 2]));
      expected[i] = add(mul(alpha, sum), mul(beta, y[i * 2]));
    }
    blas_gemv_colmajor(2, A, 3, Span<const Complex>(x, 3), 2,
                       Span<Complex>(y, 3), 2, alpha, beta);
    assert(complex_near(y[0], expected[0], kEps));
    assert(complex_near(y[2], expected[1], kEps));
    assert(y[1].re == 77 && y[1].im == 88);
  }

  // Small nonzero inputs are preserved unless approximation is requested.
  {
#if defined(FLUXPULSE_GEMV_DROP_SMALL_VALUES)
    constexpr Real expected = Real{0};
#else
    constexpr Real expected = Real{1};
#endif
    const Complex large[] = {{1e8, 0}};
    const Complex tiny[] = {{1e-8, 0}};
    const Complex unit[] = {{1, 0}};
    Complex out[] = {{0, 0}};

    // A small x entry can still give a significant product.
    blas_gemv_colmajor(1, large, 1, Span<const Complex>(tiny, 1), 1,
                       Span<Complex>(out, 1), 1, complex_one(), complex_zero());
    assert(complex_near(out[0], Complex{expected, 0}, kEps));

    // Likewise for small alpha and beta.
    blas_gemv_colmajor(1, large, 1, Span<const Complex>(unit, 1), 1,
                       Span<Complex>(out, 1), 1, Complex{1e-8, 0}, complex_zero());
    assert(complex_near(out[0], Complex{expected, 0}, kEps));
    out[0] = Complex{1e8, 0};
    blas_gemv_colmajor(1, unit, 1, Span<const Complex>(unit, 1), 1,
                       Span<Complex>(out, 1), 1, complex_zero(), Complex{1e-8, 0});
    assert(complex_near(out[0], Complex{expected, 0}, kEps));
  }

  // beta=0 must overwrite y rather than read its old NaN contents.
  {
    const Complex unit[] = {{1, 0}};
    Complex out[] = {{std::numeric_limits<double>::quiet_NaN(), 0}};
    blas_gemv_colmajor(1, unit, 1, Span<const Complex>(unit, 1), 1,
                       Span<Complex>(out, 1), 1, complex_one(), complex_zero());
    assert(complex_near(out[0], complex_one(), kEps));
  }

  // An empty matrix leaves y unchanged.
  {
    const Complex unit[] = {{1, 0}};
    Complex out[] = {{2, -3}};
    blas_gemv_colmajor(0, unit, 0, Span<const Complex>(unit, 1), 1,
                       Span<Complex>(out, 1), 1, complex_one(), complex_zero());
    assert(complex_near(out[0], Complex{2, -3}, kEps));
  }

  return 0;
}
