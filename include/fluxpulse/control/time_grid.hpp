// include/fluxpulse/control/time_grid.hpp
#pragma once

#include <cstdint>
#include <vector>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// TimeGrid supports two representations:
//
// 1) Uniform:
//    t(i) = t0 + dt * i, for i = 0..(n_points-1)
//
// 2) Explicit (non-uniform ready):
//    t(i) = tlist[i], for i = 0..(tlist.size()-1)
//
// Notes / conventions:
// - Times are in whatever units your model expects (e.g., ns, us, or normalized).
// - For explicit grids we require strictly increasing times (t[i+1] > t[i]).
// - This is not a hot-path kernel type, so it may own a std::vector for the explicit case.
struct TimeGrid {
  enum class Kind : uint8_t {
    kUniform = 0,
    kExplicit = 1,
  };

  Kind kind = Kind::kUniform;

  // Uniform representation:
  Real t0 = Real{0};
  Real dt = Real{0};
  Dim n_points = 0;

  // Explicit representation:
  // Used only when kind == kExplicit.
  std::vector<Real> tlist;

  // Constructors (factory helpers)
  static TimeGrid Uniform(Real t0_in, Real dt_in, Dim n_points_in);
  static TimeGrid Explicit(std::vector<Real> tlist_in);

  // Basic queries
  bool is_uniform() const { return kind == Kind::kUniform; }
  bool is_explicit() const { return kind == Kind::kExplicit; }

  Dim size() const;          // number of time points
  Dim num_intervals() const; // max(0, size()-1)

  // Access time points.
  // Preconditions: i < size()
  Real t(Dim i) const;

  // Access time step for interval i -> i+1.
  // Preconditions: i + 1 < size()
  Real dt_interval(Dim i) const;

  Real t_begin() const;  // first time point (precondition: size()>0)
  Real t_end() const;    // last time point  (precondition: size()>0)
  Real duration() const; // t_end - t_begin (precondition: size()>0)

  // For explicit grids: returns the internal tlist as a Span.
  // For uniform grids: returns an empty span.
  Span<const Real> explicit_tlist() const;

  // Validation (no exceptions). Useful for config loading and tests.
  // - Uniform: dt > 0, n_points > 0
  // - Explicit: tlist.size() > 0 and strictly increasing
  Status validate() const;
};

}  // namespace fluxpulse