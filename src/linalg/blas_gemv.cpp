// src/linalg/blas_gemv.cpp
#include "fluxpulse/linalg/blas_gemv.hpp"

#include <cassert>
#include <cstddef>

namespace fluxpulse {

static inline bool complex_near_zero(Complex z) noexcept {
  return abs2(z) <= kEpsilon;
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

  // y <- beta*y
  if (complex_near_zero(beta)) {
    Index yi = 0;
    for (Dim i = 0; i < n; ++i, yi += incy) y.data[as_size_t(yi)] = complex_zero();
  } else {
    Index yi = 0;
    for (Dim i = 0; i < n; ++i, yi += incy) {
      const std::size_t k = as_size_t(yi);
      y.data[k] = mul(beta, y.data[k]);
    }
  }

  if (complex_near_zero(alpha)) return;

  // Column-major GEMV: y += alpha * A * x
  Index xj_idx = 0;
  for (Dim j = 0; j < n; ++j, xj_idx += incx) {
    const Complex xj = x.data[as_size_t(xj_idx)];
    if (complex_near_zero(xj)) continue;

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