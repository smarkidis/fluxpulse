// include/fluxpulse/control/pulse_program.hpp
#pragma once

#include <vector>

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/control/pulse.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// PulseProgram bundles:
// - a TimeGrid (uniform or explicit), and
// - a set of Pulses (each pulse = channel + envelope).
//
// It provides coefficient queries u(channel, t) and u(channel, i) where i is a
// time-grid index.
//
// Phase 1 convention:
// - at most one Pulse per channel (enforced by validate()).
// - envelopes should be finalized (sorted, non-overlapping) (enforced by validate()).
struct PulseProgram {
  struct Cursor {
    // One cursor per pulse/envelope to allow fast monotonic evaluation.
    // Same indexing as `pulses`.
    std::vector<Envelope::Cursor> env;
  };

  TimeGrid grid;
  std::vector<Pulse> pulses;

  PulseProgram() = default;
  explicit PulseProgram(TimeGrid g) : grid(std::move(g)) {}

  void clear() {
    pulses.clear();
  }

  // Adds a pulse (does not validate).
  void add_pulse(const Pulse& p) { pulses.push_back(p); }
  void add_pulse(Pulse&& p) { pulses.push_back(std::move(p)); }

  Dim num_pulses() const { return static_cast<Dim>(pulses.size()); }

  // Create / reset a cursor suitable for evaluating coefficients in a solver loop.
  Cursor make_cursor() const;
  static void reset(Cursor* c);

  // Validate program invariants:
  // - grid.validate()
  // - unique channels (no duplicates)
  // - each envelope.finalize() already called successfully OR enforce by calling finalize here
  //
  // Behavior:
  // - If finalize_envelopes is true, this function will call envelope.finalize() on each pulse.
  Status validate(bool finalize_envelopes);

  // Coefficient query at a given time. Uses binary-search evaluation in envelopes.
  Real coeff(ChannelId ch, Real t) const;

  // Coefficient query at time-grid point i, using monotonic evaluation with a cursor.
  // Intended for solver loops where i increases.
  //
  // Preconditions:
  // - i < grid.size()
  // - cursor != nullptr
  // - cursor created by make_cursor() and not resized externally
  Real coeff_on_grid(ChannelId ch, Dim i, Cursor* cursor) const;

  // Convenience: coefficient for pulse index pidx at grid index i.
  // This avoids channel lookup overhead in tight loops if you iterate pulses directly.
  Real coeff_pulse_on_grid(Dim pidx, Dim i, Cursor* cursor) const;

  // Find pulse index by channel (linear search in Phase 1; can be optimized later).
  // Returns -1 if not found.
  int find_pulse_index(ChannelId ch) const;
};

}  // namespace fluxpulse