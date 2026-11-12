// include/fluxpulse/linalg/blas_gemv.hpp
#pragma once

// BLAS GEMV wrapper.
//
// Purpose:
// - Provide a single call site for matrix-vector multiply without changing
// callers.
// - Keep all BLAS/cublas types, handles, and conversions OUT of DenseOp and OUT
// of kernels.
//
// CMake selects FLUXPULSE_BLAS_BACKEND:
// - NONE: CPU loops, default, with exact-zero checks.
// - ACCELERATE: macOS CBLAS.
// - LIBSCI: Cray CBLAS via the CC wrapper.
// Vendor headers and complex-layout checks live in the implementation.
// With NONE, FLUXPULSE_GEMV_DROP_SMALL_VALUES=ON uses an approximation
// that treats alpha, beta, and x entries with abs2(z) <= kEpsilon as zero.

#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {

// y = alpha * A * x + beta * y
// A is column-major with leading dimension lda (A is n x n in Phase 1).
// incx/incy are included for future flexibility; for now DenseOp uses 1.
void blas_gemv_colmajor(Dim n, const Complex *A, Dim lda, Span<const Complex> x,
                        Index incx, Span<Complex> y, Index incy, Complex alpha,
                        Complex beta) noexcept;

} // namespace fluxpulse
