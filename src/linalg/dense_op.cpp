#include "fluxpulse/linalg/dense_op.hpp"

#include <cstddef>

#include "fluxpulse/linalg/blas_gemv.hpp"

namespace fluxpulse {

void DenseOp::resize(Dim n) {
  assert(n >= 0);
  n_ = n;
  lda_ = n;
  a_.assign(as_size_t(lda_ * n_), complex_zero());
}

Complex& DenseOp::at(Index row, Index col) noexcept {
  assert(row >= 0 && row < n_);
  assert(col >= 0 && col < n_);
  const std::size_t idx = as_size_t(col * lda_ + row);
  return a_[idx];
}

const Complex& DenseOp::at(Index row, Index col) const noexcept {
  assert(row >= 0 && row < n_);
  assert(col >= 0 && col < n_);
  const std::size_t idx = as_size_t(col * lda_ + row);
  return a_[idx];
}

void DenseOp::set_zero() noexcept {
  for (std::size_t k = 0; k < a_.size(); ++k) {
    a_[k] = complex_zero();
  }
}

void DenseOp::set_identity() noexcept {
  set_zero();
  for (Dim i = 0; i < n_; ++i) {
    at(i, i) = complex_one();
  }
}

void DenseOp::apply(Span<Complex> out, Span<const Complex> in) const noexcept {
  op_check_sizes(n_, out, in);
  // y = 1*A*x + 0*y
  gemv(out, in, complex_one(), complex_zero());
}

void DenseOp::gemv(Span<Complex> y,
                   Span<const Complex> x,
                   Complex alpha,
                   Complex beta) const noexcept {
  op_check_sizes(n_, y, x);

  // Centralized GEMV backend call (naive now; BLAS/cuBLAS later).
  // Column-major A with leading dimension lda_.
  blas_gemv_colmajor(
      n_,
      a_.data(),
      lda_,
      x,
      /*incx=*/1,
      y,
      /*incy=*/1,
      alpha,
      beta);
}

}  // namespace fluxpulse