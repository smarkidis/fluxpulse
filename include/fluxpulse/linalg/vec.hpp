#pragma once

// Motivation:
// - Provide a minimal C-like "view" over contiguous memory: (pointer + length).
// - Keep math kernels independent of std containers (CUDA-ready).
// - Avoid iterator/template-heavy APIs; just pass Span<T> into kernels.
// - Ownership stays elsewhere (e.g., std::vector on CPU, cudaMalloc buffers later).

#include <vector>
#include <cassert>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/math/complex.hpp"
#include "fluxpulse/math/real_math.hpp"

namespace fluxpulse {

using VecView = Span<Complex>;
using ConstVecView = Span<const Complex>;

struct Vec {
  std::vector<Complex> buf;

  Vec() = default;
  explicit Vec(Dim n) { resize(n); }

  Dim size() const noexcept { return static_cast<Dim>(buf.size()); }
  bool empty() const noexcept { return buf.empty(); }

  Complex* data() noexcept { return buf.data(); }
  const Complex* data() const noexcept { return buf.data(); }

  VecView view() noexcept { return make_span(data(), size()); }
  ConstVecView view() const noexcept { return make_span(data(), size()); }

  void resize(Dim n) { assert(n >= 0); buf.resize(as_size_t(n)); }
  void clear() noexcept { buf.clear(); }

  Complex& operator[](Index i) noexcept { return buf[as_size_t(i)]; }
  const Complex& operator[](Index i) const noexcept { return buf[as_size_t(i)]; }

  void fill(Complex c) noexcept { for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = c; }
  void set_zero() noexcept { fill(complex_zero()); }
};

// Kernels now take Span views:
inline void vec_copy(VecView dst, ConstVecView src) noexcept {
  assert(dst.size == src.size);
  for (Dim i = 0; i < dst.size; ++i) dst.data[as_size_t(i)] = src.data[as_size_t(i)];
}

inline void vec_set(VecView dst, Complex v) noexcept {
  for (Dim i = 0; i < dst.size; ++i) dst.data[as_size_t(i)] = v;
}

inline void vec_scale_inplace(VecView x, Real a) noexcept {
  for (Dim i = 0; i < x.size; ++i) {
    const std::size_t k = as_size_t(i);
    x.data[k] = scale(x.data[k], a);
  }
}

inline void vec_axpy_inplace(VecView y, Complex a, ConstVecView x) noexcept {
  assert(y.size == x.size);
  for (Dim i = 0; i < y.size; ++i) {
    const std::size_t k = as_size_t(i);
    y.data[k] = add(y.data[k], mul(a, x.data[k]));
  }
}

inline Complex vec_dotc(ConstVecView a, ConstVecView b) noexcept {
  assert(a.size == b.size);
  Real re = 0.0, im = 0.0;
  for (Dim i = 0; i < a.size; ++i) {
    const std::size_t k = as_size_t(i);
    const Complex prod = mul(conj(a.data[k]), b.data[k]);
    re += prod.re;
    im += prod.im;
  }
  return Complex{re, im};
}

inline Real vec_norm2(ConstVecView x) noexcept {
  Real s = 0.0;
  for (Dim i = 0; i < x.size; ++i) s += abs2(x.data[as_size_t(i)]);
  return s;
}

inline Real vec_normalize_inplace(VecView x) noexcept {
  const Real n2 = vec_norm2(ConstVecView{x.data, x.size});
  if (n2 <= kEpsilon) return 0.0;
  const Real n = real_sqrt(n2);
  vec_scale_inplace(x, 1.0 / n);
  return n;
}

}  // namespace fluxpulse