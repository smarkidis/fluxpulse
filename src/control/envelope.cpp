// src/control/envelope.cpp
#include "fluxpulse/control/envelope.hpp"

#include <algorithm>
#include <cassert>

namespace fluxpulse {
namespace {

// Computes the segment value at time t, assuming s.contains(t) is true.
inline Real eval_segment_value(const Segment& s, Real t) {
  switch (s.kind) {
    case Segment::Kind::kPiecewiseConstant: {
      return s.v0;
    }
    case Segment::Kind::kPiecewiseLinear: {
      const Real denom = (s.t1 - s.t0);
      // finalize() ensures denom > 0.
      const Real alpha = (t - s.t0) / denom;
      return s.v0 + alpha * (s.v1 - s.v0);
    }
    default: {
      // Unknown kind: treat as 0 (should not happen).
      return Real{0};
    }
  }
}

}  // namespace

Status Envelope::finalize() {
  // Sort by segment start time.
  std::sort(segments.begin(), segments.end(),
            [](const Segment& a, const Segment& b) { return a.t0 < b.t0; });

  // Validate each segment.
  for (size_t i = 0; i < segments.size(); ++i) {
    const Segment& s = segments[i];
    if (!(s.t1 > s.t0)) {
      return Error(ErrorCode::kInvalidArgument, "Envelope: segment must have t1 > t0");
    }
    if (i > 0) {
      const Segment& prev = segments[i - 1];
      // Enforce non-overlap. Gaps are allowed.
      if (s.t0 < prev.t1) {
        return Error(ErrorCode::kInvalidArgument, "Envelope: segments overlap in time");
      }
    }
  }
  return Ok();
}

Real Envelope::eval(Real t) const {
  if (segments.empty()) {
    return Real{0};
  }

  // Find first segment with t0 > t. Candidate is the previous one.
  // Equivalent to upper_bound on t0.
  auto it = std::upper_bound(
      segments.begin(), segments.end(), t,
      [](Real x, const Segment& s) { return x < s.t0; });

  if (it == segments.begin()) {
    return Real{0};
  }

  const Segment& s = *(it - 1);
  if (!s.contains(t)) {
    return Real{0};
  }
  return eval_segment_value(s, t);
}

Real Envelope::eval_monotonic(Real t, Cursor* cursor) const {
  assert(cursor != nullptr);

  if (segments.empty()) {
    cursor->seg_index = 0;
    return Real{0};
  }

  Dim idx = cursor->seg_index;
  const Dim n = static_cast<Dim>(segments.size());
  if (idx < 0) idx = 0;
  if (idx > n) idx = n;

  // Advance past segments that ended at or before t.
  while (idx < n) {
    const Segment& s = segments[static_cast<size_t>(idx)];
    if (t < s.t1) {
      break;
    }
    ++idx;
  }
  cursor->seg_index = idx;

  if (idx >= n) {
    return Real{0};
  }

  const Segment& s = segments[static_cast<size_t>(idx)];

  // If we're in a gap before the next segment starts, value is 0.
  if (!s.contains(t)) {
    return Real{0};
  }

  return eval_segment_value(s, t);
}

Real Envelope::eval_on_grid(Dim i, const TimeGrid& grid, Cursor* cursor) const {
  // grid.t(i) asserts i < grid.size() inside TimeGrid::t
  const Real ti = grid.t(i);
  return eval_monotonic(ti, cursor);
}

}  // namespace fluxpulse