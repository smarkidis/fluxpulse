// include/fluxpulse/linalg/blas_gemv.hpp
#pragma once

// BLAS GEMV wrapper (placeholder).
//
// Purpose:
// - Provide a single call site for matrix-vector multiply that can later be backed by
//   CBLAS or cuBLAS without changing operator/solver code.
// - Keep all BLAS/cublas types, handles, and conversions OUT of DenseOp and OUT of kernels.
//
// Current behavior:
// - Always uses a naive implementation (no external BLAS dependency).
//
// Future:
// - If FLUXPULSE_ENABLE_CBLAS is enabled, this wrapper can call cblas_*gemv.
// - If FLUXPULSE_ENABLE_CUBLAS is enabled, this wrapper can call cublas*gemv.
// - Any complex-type conversions (fluxpulse::Complex <-> vendor complex) live here.

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {

// y = alpha * A * x + beta * y
// A is column-major with leading dimension lda (A is n x n in Phase 1).
// incx/incy are included for future flexibility; for now DenseOp uses 1.
void blas_gemv_colmajor(Dim n,
                        const Complex* A,
                        Dim lda,
                        Span<const Complex> x,
                        Index incx,
                        Span<Complex> y,
                        Index incy,
                        Complex alpha,
                        Complex beta) noexcept;

}  // namespace fluxpulse