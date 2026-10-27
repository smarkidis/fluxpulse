// include/fluxpulse/core/span.hpp
#pragma once

#include <cstddef>
#include <cassert>

#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// Minimal pointer+length view type.
// - C-like: just data pointer + size.
// - Works for T and const T (use Span<const T>).
// - Header-only leaf utility.
//
// Notes:
// - size is Dim to match the rest of the codebase (signed).
// - No bounds checks in operator[] (Debug asserts only).

template <class T>
struct Span {
  T* data = nullptr;
  Dim size = 0;

  FLUXPULSE_HD constexpr Span() noexcept = default;
  FLUXPULSE_HD constexpr Span(T* p, Dim n) noexcept : data(p), size(n) {}

  FLUXPULSE_HD constexpr bool empty() const noexcept { return size <= 0 || data == nullptr; }

  FLUXPULSE_HD T& operator[](Index i) const noexcept {
    assert(i >= 0);
    assert(i < size);
    return data[as_size_t(i)];
  }
};

template <class T>
FLUXPULSE_HD constexpr Span<T> make_span(T* p, Dim n) noexcept {
  return Span<T>(p, n);
}

}  // namespace fluxpulse