// include/fluxpulse/linalg/dense_op.hpp
#pragma once

// Dense operator (square matrix) stored in **column-major** order.
//
// Design decisions (BLAS-ready):
// 1) Store column-major and expose data() + lda().
// 2) Provide gemv(alpha, beta): y = alpha*A*x + beta*y.
// 3) Keep BLAS/cublas types and conversions out of this class (dedicated wrapper later).
//
// Notes:
// - Uses POD Complex (fluxpulse::Complex), not std::complex.
// - Uses Span<T> for kernel-like APIs.
// - No exceptions; Debug asserts enforce preconditions.

#include <vector>
#include <cassert>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/linalg/op.hpp"

namespace fluxpulse {

enum class MatrixLayout : std::uint8_t {
  kColMajor = 0
};

// Dense square operator A (n x n), stored column-major.
// Element A(row, col) is stored at data()[col*lda + row].
struct DenseOp final : public Op {
  DenseOp() = default;
  explicit DenseOp(Dim n) { resize(n); }

  // Resize to n x n and zero-fill.
  void resize(Dim n);

  // Op interface
  Dim dim() const noexcept override { return n_; }
  void apply(Span<Complex> out, Span<const Complex> in) const noexcept override;

  // BLAS-shaped interface: y = alpha*A*x + beta*y
  void gemv(Span<Complex> y,
            Span<const Complex> x,
            Complex alpha,
            Complex beta) const noexcept;

  // Matrix metadata (for BLAS wrappers later)
  MatrixLayout layout() const noexcept { return MatrixLayout::kColMajor; }
  Dim lda() const noexcept { return lda_; }

  // Raw access to storage (column-major).
  Complex* data() noexcept { return a_.data(); }
  const Complex* data() const noexcept { return a_.data(); }

  // Element access (row, col). Unchecked in Release; Debug asserts only.
  Complex& at(Index row, Index col) noexcept;
  const Complex& at(Index row, Index col) const noexcept;

  void set(Index row, Index col, Complex v) noexcept { at(row, col) = v; }

  void set_zero() noexcept;
  void set_identity() noexcept;

private:
  Dim n_ = 0;
  Dim lda_ = 0;                 // leading dimension (stride between column starts), in elements
  std::vector<Complex> a_{};    // size = lda_ * n_ (column-major)
};

}  // namespace fluxpulse