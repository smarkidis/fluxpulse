// src/control/pulse_program.cpp
#include "fluxpulse/control/pulse_program.hpp"

#include <cassert>

namespace fluxpulse {

PulseProgram::Cursor PulseProgram::make_cursor() const {
  Cursor c;
  c.env.resize(pulses.size());
  for (size_t i = 0; i < c.env.size(); ++i) {
    Envelope::reset(&c.env[i]);
  }
  return c;
}

void PulseProgram::reset(Cursor* c) {
  // Static: cannot access PulseProgram::pulses.
  // Only reset the cursor entries it already has.
  assert(c != nullptr);
  for (size_t i = 0; i < c->env.size(); ++i) {
    Envelope::reset(&c->env[i]);
  }
}

int PulseProgram::find_pulse_index(ChannelId ch) const {
  for (size_t i = 0; i < pulses.size(); ++i) {
    if (pulses[i].channel == ch) return static_cast<int>(i);
  }
  return -1;
}

Status PulseProgram::validate(bool finalize_envelopes) {
  FLUXPULSE_RETURN_IF_ERROR(grid.validate());

  // Phase 1: one pulse per channel
  for (size_t i = 0; i < pulses.size(); ++i) {
    const ChannelId ch = pulses[i].channel;
    if (!ch.valid()) {
      return Error(ErrorCode::kInvalidArgument, "PulseProgram: invalid ChannelId (value=0)");
    }
    for (size_t j = i + 1; j < pulses.size(); ++j) {
      if (pulses[j].channel == ch) {
        return Error(ErrorCode::kInvalidArgument,
                     "PulseProgram: duplicate channel (Phase 1 allows one pulse per channel)");
      }
    }
  }

  if (finalize_envelopes) {
    for (size_t i = 0; i < pulses.size(); ++i) {
      Status st = pulses[i].envelope.finalize();
      if (!st.ok()) return Annotate(st, "PulseProgram: envelope.finalize");
    }
  }

  return Ok();
}

Real PulseProgram::coeff(ChannelId ch, Real t) const {
  const int idx = find_pulse_index(ch);
  if (idx < 0) return Real{0};
  return pulses[static_cast<size_t>(idx)].envelope.eval(t);
}

Real PulseProgram::coeff_on_grid(ChannelId ch, Dim i, Cursor* cursor) const {
  assert(cursor != nullptr);
  assert(i >= 0 && i < grid.size());

  const int idx = find_pulse_index(ch);
  if (idx < 0) return Real{0};

  const size_t uidx = static_cast<size_t>(idx);
  if (cursor->env.size() != pulses.size()) return Real{0};
  return pulses[uidx].envelope.eval_on_grid(i, grid, &cursor->env[uidx]);
}

Real PulseProgram::coeff_pulse_on_grid(Dim pidx, Dim i, Cursor* cursor) const {
  assert(cursor != nullptr);
  assert(i >= 0 && i < grid.size());
  assert(pidx >= 0 && static_cast<size_t>(pidx) < pulses.size());

  if (cursor->env.size() != pulses.size()) return Real{0};
  const size_t uidx = static_cast<size_t>(pidx);
  return pulses[uidx].envelope.eval_on_grid(i, grid, &cursor->env[uidx]);
}

}  // namespace fluxpulse