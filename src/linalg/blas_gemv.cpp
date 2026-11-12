// src/linalg/blas_gemv.cpp
#include "fluxpulse/linalg/blas_gemv.hpp"

#include <cassert>
#include <cstddef>

#if defined(FLUXPULSE_USE_ACCELERATE) || defined(FLUXPULSE_USE_CBLAS)
#include <limits>
#endif

#if defined(FLUXPULSE_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(FLUXPULSE_USE_CBLAS)
#include <cblas.h>
#endif

namespace fluxpulse {

#if defined(FLUXPULSE_USE_ACCELERATE) || defined(FLUXPULSE_USE_CBLAS)
// CBLAS complex arguments are interleaved real/imaginary doubles.
static_assert(sizeof(Real) == sizeof(double));
static_assert(sizeof(Complex) == 2 * sizeof(double));
static_assert(offsetof(Complex, re) == 0);
static_assert(offsetof(Complex, im) == sizeof(double));
#if defined(FLUXPULSE_USE_ACCELERATE)
// Accelerate's newer C++ interface requires typed complex pointers.
using BlasComplex = __LAPACK_double_complex;
static_assert(sizeof(BlasComplex) == sizeof(Complex));
static_assert(alignof(Complex) >= alignof(BlasComplex));
#else
using BlasComplex = Complex;
#endif
#endif

static inline bool gemv_is_zero(Complex z) noexcept {
#if defined(FLUXPULSE_GEMV_DROP_SMALL_VALUES)
  return abs2(z) <= kEpsilon;
#else
  return z.re == Real{0} && z.im == Real{0};
#endif
}

void blas_gemv_colmajor(Dim n,
                        const Complex* A,
                        Dim lda,
                        Span<const Complex> x,
                        Index incx,
                        Span<Complex> y,
                        Index incy,
                        Complex alpha,
                        Complex beta) noexcept {
  assert(n >= 0);
  assert(lda >= n);
  assert(A != nullptr);
  assert(x.data != nullptr);
  assert(y.data != nullptr);
  assert(incx > 0);
  assert(incy > 0);

  if (n == 0)
    return;

#if defined(FLUXPULSE_USE_ACCELERATE) || defined(FLUXPULSE_USE_CBLAS)
  // Stay within the 32-bit BLAS interface range; retain the native-width
  // implementation below for dimensions/strides outside that range.
  constexpr auto max_blas_int = std::numeric_limits<int>::max();
  if (n <= max_blas_int && lda <= max_blas_int && incx <= max_blas_int &&
      incy <= max_blas_int) {
    cblas_zgemv(
        CblasColMajor, CblasNoTrans, static_cast<int>(n), static_cast<int>(n),
        reinterpret_cast<const BlasComplex *>(&alpha),
        reinterpret_cast<const BlasComplex *>(A), static_cast<int>(lda),
        reinterpret_cast<const BlasComplex *>(x.data), static_cast<int>(incx),
        reinterpret_cast<const BlasComplex *>(&beta),
        reinterpret_cast<BlasComplex *>(y.data), static_cast<int>(incy));
    return;
  }
#endif

  // y <- beta*y
  if (gemv_is_zero(beta)) {
    Index yi = 0;
    for (Dim i = 0; i < n; ++i, yi += incy) y.data[as_size_t(yi)] = complex_zero();
  } else {
    Index yi = 0;
    for (Dim i = 0; i < n; ++i, yi += incy) {
      const std::size_t k = as_size_t(yi);
      y.data[k] = mul(beta, y.data[k]);
    }
  }

  if (gemv_is_zero(alpha)) return;

  // Column-major GEMV: y += alpha * A * x
  Index xj_idx = 0;
  for (Dim j = 0; j < n; ++j, xj_idx += incx) {
    const Complex xj = x.data[as_size_t(xj_idx)];
    if (gemv_is_zero(xj)) continue;

    const Complex t = mul(alpha, xj);
    const Complex* col = A + as_size_t(j * lda);

    Index yi = 0;
    for (Dim i = 0; i < n; ++i, yi += incy) {
      const std::size_t yk = as_size_t(yi);
      y.data[yk] = add(y.data[yk], mul(col[as_size_t(i)], t));
    }
  }
}

}  // namespace fluxpulse