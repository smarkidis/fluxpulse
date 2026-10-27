// include/fluxpulse/control/envelope.hpp
#pragma once

#include <cstdint>
#include <vector>

#include "fluxpulse/control/segment.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// Envelope: a (typically) piecewise-defined real-valued function u(t).
//
// Phase 1 convention (recommended):
// - Segments are sorted by t0.
// - Segments do not overlap (prev.t1 <= next.t0).
// - Outside all segments, u(t) = 0.
//
// This representation is convenient for pulse scheduling and for exporting a
// timeline (e.g., Perfetto tracks per channel).
struct Envelope {
  // A mutable cursor that allows O(1) amortized evaluation when time is
  // non-decreasing (typical solver loops).
  struct Cursor {
    Dim seg_index = 0;
  };

  std::vector<Segment> segments;

  // Adds a segment (does not sort/validate).
  void add(const Segment& s) { segments.push_back(s); }

  void clear() { segments.clear(); }

  Dim size() const { return static_cast<Dim>(segments.size()); }

  // Sorts by t0 and validates:
  // - each segment has t1 > t0
  // - strictly non-overlapping in time (prev.t1 <= next.t0)
  // Returns Status::ok() on success.
  Status finalize();

  // Evaluate u(t) using binary search (log N).
  // Requires finalize() has been called successfully (sorted segments).
  Real eval(Real t) const;

  // Evaluate u(t) assuming t is non-decreasing across calls using a Cursor.
  // This avoids binary search; it advances the cursor as needed.
  // Requires finalize() has been called successfully (sorted segments).
  Real eval_monotonic(Real t, Cursor* cursor) const;

  // Convenience for evaluating on a TimeGrid at index i:
  // - uses grid.t(i)
  // - uses monotonic evaluation (recommended in solver loops)
  Real eval_on_grid(Dim i, const TimeGrid& grid, Cursor* cursor) const;

  // Resets cursor to the beginning (useful before a solver time loop).
  static void reset(Cursor* cursor) { cursor->seg_index = 0; }
};

}  // namespace fluxpulse