// include/fluxpulse/linalg/op.hpp
#pragma once

// Operator interface.
//
// Motivation:
// - Provide a small, explicit abstraction for linear operators used by the simulator
//   (Hamiltonians, collapse operators, embedded tensor ops, etc.).
// - Keep kernels C-like: apply() takes Span pointers, not std containers.
// - Allow different operator implementations (dense, diagonal, sparse later) behind
//   the same call shape, with minimal inheritance.
//
// Notes:
// - This interface is HOST-side. CUDA backends will typically launch kernels directly
//   and may not use virtual dispatch on device.
// - Error handling: no exceptions. Preconditions are documented and checked with
//   assert() in Debug builds.

#include <cassert>

#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/math/complex.hpp"

namespace fluxpulse {

// Square linear operator acting on state vectors of size dim().
struct Op {
  virtual ~Op() = default;

  // Dimension of the (square) operator. State vectors must have this length.
  virtual Dim dim() const noexcept = 0;

  // out <- A * in
  //
  // Preconditions:
  // - out.size == dim()
  // - in.size  == dim()
  // - out.data and in.data must not alias (unless an implementation explicitly documents support).
  virtual void apply(Span<Complex> out, Span<const Complex> in) const noexcept = 0;
};

// Helper for Debug precondition checking in derived classes.
//
// Note: In Release builds (NDEBUG), assert() compiles out and the parameters would otherwise
// appear unused under -Wall/-Wextra. We explicitly mark them maybe_unused and cast-to-void.
inline void op_check_sizes([[maybe_unused]] Dim dim,
                           [[maybe_unused]] Span<Complex> out,
                           [[maybe_unused]] Span<const Complex> in) noexcept {
#ifndef NDEBUG
  assert(out.data != nullptr);
  assert(in.data != nullptr);
  assert(out.size == dim);
  assert(in.size == dim);
#else
  (void)dim;
  (void)out;
  (void)in;
#endif
}

}  // namespace fluxpulse
