// src/control/time_grid.cpp
#include "fluxpulse/control/time_grid.hpp"

#include <cassert>

namespace fluxpulse {

TimeGrid TimeGrid::Uniform(Real t0_in, Real dt_in, Dim n_points_in) {
  TimeGrid g;
  g.kind = Kind::kUniform;
  g.t0 = t0_in;
  g.dt = dt_in;
  g.n_points = n_points_in;
  g.tlist.clear();
  return g;
}

TimeGrid TimeGrid::Explicit(std::vector<Real> tlist_in) {
  TimeGrid g;
  g.kind = Kind::kExplicit;
  g.t0 = Real{0};
  g.dt = Real{0};
  g.n_points = 0;
  g.tlist = std::move(tlist_in);
  return g;
}

Dim TimeGrid::size() const {
  if (kind == Kind::kUniform) {
    return n_points;
  }
  return static_cast<Dim>(tlist.size());
}

Dim TimeGrid::num_intervals() const {
  const Dim n = size();
  return (n > 0) ? (n - 1) : 0;
}

Real TimeGrid::t(Dim i) const {
  const Dim n = size();
  assert(i < n);
  (void)n;  // silence -Wunused-variable when asserts compile out
  if (kind == Kind::kUniform) {
    return t0 + dt * static_cast<Real>(i);
  }
  return tlist[static_cast<size_t>(i)];
}

Real TimeGrid::dt_interval(Dim i) const {
  const Dim n = size();
  assert(i + 1 < n);
  (void)n;  // silence -Wunused-variable when asserts compile out
  if (kind == Kind::kUniform) {
    return dt;
  }
  const Real t0i = t(i);
  const Real t1i = t(i + 1);
  return t1i - t0i;
}

Real TimeGrid::t_begin() const {
  const Dim n = size();
  assert(n > 0);
  (void)n;  // silence -Wunused-variable when asserts compile out
  if (kind == Kind::kUniform) {
    return t0;
  }
  return tlist.front();
}

Real TimeGrid::t_end() const {
  const Dim n = size();
  assert(n > 0);
  (void)n;  // silence -Wunused-variable when asserts compile out
  if (kind == Kind::kUniform) {
    // last point: t0 + dt*(n-1)
    return t0 + dt * static_cast<Real>(n - 1);
  }
  return tlist.back();
}

Real TimeGrid::duration() const {
  const Dim n = size();
  assert(n > 0);
  (void)n;  // silence -Wunused-variable when asserts compile out
  return t_end() - t_begin();
}

Span<const Real> TimeGrid::explicit_tlist() const {
  if (kind != Kind::kExplicit) {
    return Span<const Real>(nullptr, 0);
  }
  if (tlist.empty()) {
    return Span<const Real>(nullptr, 0);
  }
  return Span<const Real>(tlist.data(), static_cast<Dim>(tlist.size()));
}

Status TimeGrid::validate() const {
  if (kind == Kind::kUniform) {
    if (!(dt > Real{0})) {
      return Error(ErrorCode::kInvalidArgument, "TimeGrid uniform: dt must be > 0");
    }
    if (n_points <= 0) {
      return Error(ErrorCode::kInvalidArgument, "TimeGrid uniform: n_points must be > 0");
    }
    return Ok();
  }

  // Explicit
  if (tlist.empty()) {
    return Error(ErrorCode::kInvalidArgument, "TimeGrid explicit: tlist must be non-empty");
  }
  // Strictly increasing (avoids zero/negative dt).
  for (size_t i = 0; i + 1 < tlist.size(); ++i) {
    const Real a = tlist[i];
    const Real b = tlist[i + 1];
    if (!(b > a)) {
      return Error(ErrorCode::kInvalidArgument,
                   "TimeGrid explicit: tlist must be strictly increasing");
    }
  }
  return Ok();
}

}  // namespace fluxpulse
