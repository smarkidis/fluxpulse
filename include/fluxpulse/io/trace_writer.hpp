// include/fluxpulse/io/trace_writer.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/core/span.hpp"

namespace fluxpulse {
namespace io {

// Perfetto can load the legacy "Chrome JSON trace" format.
// We emit timestamps in *nanoseconds* to preserve sub-microsecond pulse structure.
//
// We still accept a config knob `time_scale_to_us` (simulation time -> microseconds),
// but we convert internally to ns:
//   ts_ns  = round(t * time_scale_to_us * 1000)
//   dur_ns = round((t1-t0) * time_scale_to_us * 1000)
//
// This prevents the "all events at ts=0" problem when pulses are < 1 us.

struct TraceArg {
  std::string name;
  Real value = Real{0};
};

struct TraceEvent {
  std::string name;   // shown in UI
  std::string cat;    // category
  char ph = 'X';      // 'X' slice, 'C' counter, 'i' instant
  char scope = 0;   // for instant events: 't' (thread), 'p' (process), 'g' (global); 0 => omit
  uint64_t ts_ns = 0; // timestamp in ns
  uint64_t dur_ns = 0;// duration in ns (only for 'X')
  int pid = 1;
  int tid = 0;

  // Chrome trace events support args as a JSON object.
  // Backward compat: the old single-arg fields are still supported.
  std::string arg_name;  // e.g. "amp" or "u"
  Real arg_value = Real{0};

  // Additional args (used for multi-series counters such as MCWF diagnostics).
  std::vector<TraceArg> args;
};

// Write events to a Chrome JSON trace file.
Status write_chrome_trace_json(const std::string& path, const std::vector<TraceEvent>& events);

// Pulse visualization (recommended)
// - Slice events per envelope segment (per-channel thread track).
Status append_pulse_segment_events(const PulseProgram& program,
                                  Real time_scale_to_us,
                                  const std::string& name_prefix,
                                  std::vector<TraceEvent>* out_events);

// Optional: Counter events (amplitude samples) per channel.
// - stride: emit one sample every `stride` grid points (and always include last point).
Status append_pulse_counter_events(const PulseProgram& program,
                                  Dim stride,
                                  Real time_scale_to_us,
                                  const std::string& name_prefix,
                                  std::vector<TraceEvent>* out_events);

// Always-useful: A solver/run span so traces are not empty (e.g. T1 has no pulses).
// tid is fixed to a "solver" thread (tid=9999).
Status append_run_span_event(const PulseProgram& program,
                            Real time_scale_to_us,
                            const std::string& name,
                            std::vector<TraceEvent>* out_events);

// MCWF diagnostics counters (optional).
//
// Expected input shapes:
// - tlist:       length N
// - norm2:       length N   (typically ||psi_propagated||^2 before renorm; last entry can be final ||psi||^2)
// - p_total:     length N   (jump prob for step i->i+1; last entry usually 0)
// - jumped:      length N   (0/1; last entry usually 0)
// - jump_index:  length N   (-1 if none; last entry usually -1)
//
// Emitted as a single counter track per tid, with multiple series:
//   { "norm2": ..., "p_total": ..., "jumped": ..., "jump_index": ... }
//
// Default tid is 9999 (solver), but you can pass a different tid (e.g., per-trajectory track).
Status append_mcwf_diagnostic_counter_events(Span<const Real> tlist,
                                            Span<const Real> norm2,
                                            Span<const Real> p_total,
                                            Span<const int32_t> jumped,
                                            Span<const int32_t> jump_index,
                                            Real time_scale_to_us,
                                            const std::string& track_name,
                                            int tid,
                                            std::vector<TraceEvent>* out_events);

// MCWF jump instant events (never downsampled).
//
// Expected input shapes:
// - t_jump:      length M (jump times)
// - jump_index:  length M (collapse op index for each jump)
//
// Emitted as instant events (ph:"i") on a dedicated track (default tid=9998).
Status append_mcwf_jump_instant_events(Span<const Real> t_jump,
                                      Span<const int32_t> jump_index,
                                      Real time_scale_to_us,
                                      const std::string& track_name,
                                      int tid,
                                      std::vector<TraceEvent>* out_events);

}  // namespace io
}  // namespace fluxpulse
