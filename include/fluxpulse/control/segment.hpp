// include/fluxpulse/control/segment.hpp
#pragma once

#include <cstdint>

#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// A segment defines an envelope contribution over a half-open interval [t0, t1).
// Convention:
// - Active for t in [t0, t1)
// - Inactive for t < t0 or t >= t1
//
// Times are in the same units as the TimeGrid/model.
struct Segment {
  enum class Kind : uint8_t {
    kPiecewiseConstant = 0,  // ZOH: Zero-Order Hold (step)
    kPiecewiseLinear = 1,    // Linear interpolation between endpoints
  };

  Kind kind = Kind::kPiecewiseConstant;

  // Time interval:
  Real t0 = Real{0};
  Real t1 = Real{0};

  // Parameters:
  // - PiecewiseConstant: v0 is the constant value. (v1 ignored)
  // - PiecewiseLinear:   u(t0)=v0, u(t1)=v1 linearly.
  Real v0 = Real{0};
  Real v1 = Real{0};

  static Segment PiecewiseConstant(Real t0_in, Real t1_in, Real value) {
    Segment s;
    s.kind = Kind::kPiecewiseConstant;
    s.t0 = t0_in;
    s.t1 = t1_in;
    s.v0 = value;
    s.v1 = Real{0};
    return s;
  }

  static Segment PiecewiseLinear(Real t0_in, Real t1_in, Real v0_in, Real v1_in) {
    Segment s;
    s.kind = Kind::kPiecewiseLinear;
    s.t0 = t0_in;
    s.t1 = t1_in;
    s.v0 = v0_in;
    s.v1 = v1_in;
    return s;
  }

  bool contains(Real t) const {
    return (t >= t0) && (t < t1);
  }
};

}  // namespace fluxpulse