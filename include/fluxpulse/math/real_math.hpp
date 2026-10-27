// include/fluxpulse/math/real_math.hpp
#pragma once

// Centralized real-valued math wrappers.
// Motivation:
// - Keep kernels and math headers (e.g., complex.hpp) free of <cmath> / std:: calls.
// - Provide a single place to adjust implementation for CUDA builds later.
// - Keep the API C-like: plain functions, no exceptions, no templates.

#include <math.h>  // ::sqrt, ::fabs, ::log, ::exp, ::sin, ::cos

#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// NOTE: These wrappers intentionally call the global C math functions (::sqrt, etc.)
// rather than std::sqrt. This generally plays nicer with CUDA device compilation.

FLUXPULSE_HD inline Real real_sqrt(Real x) noexcept {
  // Precondition: x >= 0 for real result.
  return static_cast<Real>(::sqrt(x));
}

FLUXPULSE_HD inline Real real_abs(Real x) noexcept {
  return static_cast<Real>(::fabs(x));
}

FLUXPULSE_HD inline Real real_log(Real x) noexcept {
  // Precondition: x > 0.
  return static_cast<Real>(::log(x));
}

FLUXPULSE_HD inline Real real_exp(Real x) noexcept {
  return static_cast<Real>(::exp(x));
}

FLUXPULSE_HD inline Real real_sin(Real x) noexcept {
  return static_cast<Real>(::sin(x));
}

FLUXPULSE_HD inline Real real_cos(Real x) noexcept {
  return static_cast<Real>(::cos(x));
}

}  // namespace fluxpulse