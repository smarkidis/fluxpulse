// src/io/trace_writer.cpp
#include "fluxpulse/io/trace_writer.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

#include "fluxpulse/control/channel_id.hpp"

namespace fluxpulse {
namespace io {
namespace {

constexpr int kDefaultPid = 1;
constexpr int kSolverTid = 9999;
constexpr int kJumpsTid = 9998;

inline uint64_t to_ns(Real t, Real scale_to_us) {
  // Convert simulation time -> microseconds (scale_to_us) -> nanoseconds (*1000).
  const double v = static_cast<double>(t) * static_cast<double>(scale_to_us) * 1000.0;
  if (v <= 0.0) return 0;
  // round-to-nearest
  return static_cast<uint64_t>(v + 0.5);
}

inline uint64_t dur_ns_from(Real t0, Real t1, Real scale_to_us) {
  const uint64_t a = to_ns(t0, scale_to_us);
  const uint64_t b = to_ns(t1, scale_to_us);
  if (b > a) return b - a;
  // If time interval is non-zero but rounds to 0ns, force a visible 1ns slice.
  if (t1 > t0) return 1;
  return 0;
}

inline std::string channel_name_from_tid(int tid) {
  // Solver track
  if (tid == kSolverTid) return "solver";

  // Jump instant events track
  if (tid == kJumpsTid) return "jumps";

  // Phase-1 multi-qubit channel convention:
  //   DriveX_i => id = 1 + 3*i
  //   DriveY_i => id = 2 + 3*i
  //   DriveZ_i => id = 3 + 3*i
  // For i==0, keep historical names "drive_x", "drive_y", "drive_z".
  if (tid > 0) {
    const int base0 = tid - 1;
    if (base0 >= 0) {
      const int axis_mod = base0 % 3;     // 0,1,2
      const int q = base0 / 3;           // qubit index
      if (q >= 0) {
        const char* axis = (axis_mod == 0) ? "x" : (axis_mod == 1) ? "y" : "z";
        std::ostringstream ss;
        if (q == 0) {
          ss << "drive_" << axis;
        } else {
          ss << "drive_" << axis << q;
        }
        return ss.str();
      }
    }
  }

  std::ostringstream ss;
  ss << "tid_" << tid;
  return ss.str();
}

inline void write_json_string(std::ostream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\b': os << "\\b"; break;
      case '\f': os << "\\f"; break;
      case '\n': os << "\\n"; break;
      case '\r': os << "\\r"; break;
      case '\t': os << "\\t"; break;
      default: os << c; break;
    }
  }
  os << '"';
}

inline Status open_out(std::ofstream* f, const std::string& path) {
  f->open(path, std::ios::out | std::ios::binary);
  if (!f->good()) return Error(ErrorCode::kIo, "Failed to open trace output file");
  return Ok();
}

inline void sort_events(std::vector<TraceEvent>* events) {
  std::sort(events->begin(), events->end(),
            [](const TraceEvent& a, const TraceEvent& b) {
              if (a.pid != b.pid) return a.pid < b.pid;
              if (a.tid != b.tid) return a.tid < b.tid;
              if (a.ts_ns != b.ts_ns) return a.ts_ns < b.ts_ns;
              if (a.ph != b.ph) return a.ph < b.ph;
              return a.name < b.name;
            });
}

}  // namespace

Status write_chrome_trace_json(const std::string& path, const std::vector<TraceEvent>& events_in) {
  std::ofstream f;
  FLUXPULSE_RETURN_IF_ERROR(open_out(&f, path));

  std::vector<TraceEvent> events = events_in;
  sort_events(&events);

  // Collect tids for thread_name metadata
  std::set<int> tids;
  for (const TraceEvent& e : events) tids.insert(e.tid);

  f << "{\n";
  f << "  \"traceEvents\": [\n";

  bool first = true;
  auto comma = [&]() {
    if (!first) f << ",\n";
    first = false;
  };

  // ---- process metadata ----
  comma();
  f << "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":1,\"args\":{\"name\":\"fluxpulse\"}}";

  // ---- thread metadata ----
  for (int tid : tids) {
    comma();
    f << "    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":" << tid
      << ",\"args\":{\"name\":";
    write_json_string(f, channel_name_from_tid(tid));
    f << "}}";
  }

  // ---- actual events ----
  for (const TraceEvent& e : events) {
    comma();
    f << "    {";
    f << "\"name\": "; write_json_string(f, e.name);
    f << ", \"cat\": "; write_json_string(f, e.cat.empty() ? std::string("fluxpulse") : e.cat);
    f << ", \"ph\": \"" << e.ph << "\"";
    f << ", \"ts\": " << e.ts_ns;
    f << ", \"pid\": " << e.pid;
    f << ", \"tid\": " << e.tid;

    if (e.ph == 'X') f << ", \"dur\": " << e.dur_ns;

    // Instant events need a scope (thread/process/global). Default to thread scope.
    if (e.ph == 'i' || e.ph == 'I') {
      const char s = (e.scope != 0) ? e.scope : 't';
      f << ", \"s\": \"" << s << "\"";
    }

    // args
    f << ", \"args\": {";
    bool arg_first = true;
    auto arg_comma = [&]() {
      if (!arg_first) f << ", ";
      arg_first = false;
    };

    if (!e.arg_name.empty()) {
      arg_comma();
      write_json_string(f, e.arg_name);
      f << ": " << e.arg_value;
    }
    for (const TraceArg& a : e.args) {
      arg_comma();
      write_json_string(f, a.name);
      f << ": " << a.value;
    }
    f << "}";

    f << "}";
  }

  f << "\n  ],\n";
  f << "  \"displayTimeUnit\": \"ns\"\n";
  f << "}\n";

  if (!f.good()) return Error(ErrorCode::kIo, "write_chrome_trace_json: write failed");
  return Ok();
}

Status append_pulse_segment_events(const PulseProgram& program,
                                  Real time_scale_to_us,
                                  const std::string& name_prefix,
                                  std::vector<TraceEvent>* out_events) {
  if (!out_events) return Error(ErrorCode::kInvalidArgument, "append_pulse_segment_events: out_events is null");

  // One track per channel => tid = channel.value
  for (size_t pidx = 0; pidx < program.pulses.size(); ++pidx) {
    const Pulse& p = program.pulses[pidx];
    const int tid = static_cast<int>(p.channel.value);
    const std::string ch = channel_name_from_tid(tid);

    for (size_t sidx = 0; sidx < p.envelope.segments.size(); ++sidx) {
      const Segment& s = p.envelope.segments[sidx];

      TraceEvent e;
      e.pid = kDefaultPid;
      e.tid = tid;
      e.cat = "pulse";
      e.ph = 'X';

      // More descriptive name: "pulse:drive_x1#0"
      {
        std::ostringstream ss;
        ss << name_prefix << ch << "#" << sidx;
        e.name = ss.str();
      }

      e.ts_ns = to_ns(s.t0, time_scale_to_us);
      e.dur_ns = dur_ns_from(s.t0, s.t1, time_scale_to_us);

      // Helpful tooltip: segment amplitude start
      e.arg_name = "amp";
      e.arg_value = s.v0;

      out_events->push_back(std::move(e));
    }
  }

  return Ok();
}

Status append_pulse_counter_events(const PulseProgram& program,
                                  Dim stride,
                                  Real time_scale_to_us,
                                  const std::string& name_prefix,
                                  std::vector<TraceEvent>* out_events) {
  if (!out_events) return Error(ErrorCode::kInvalidArgument, "append_pulse_counter_events: out_events is null");
  if (stride <= 0) stride = 1;

  const Dim n_points = program.grid.size();
  if (n_points <= 0) return Error(ErrorCode::kInvalidArgument, "append_pulse_counter_events: empty time grid");

  // Cursor for monotonic evaluation (fast + deterministic)
  PulseProgram::Cursor cursor = program.make_cursor();

  auto emit_at = [&](Dim i) -> Status {
    const Real t = program.grid.t(i);
    const uint64_t ts = to_ns(t, time_scale_to_us);

    for (Dim pidx = 0; pidx < static_cast<Dim>(program.pulses.size()); ++pidx) {
      const Pulse& p = program.pulses[static_cast<size_t>(pidx)];
      const int tid = static_cast<int>(p.channel.value);
      const std::string ch = channel_name_from_tid(tid);

      const Real u = program.coeff_pulse_on_grid(pidx, i, &cursor);

      TraceEvent e;
      e.pid = kDefaultPid;
      e.tid = tid;
      e.cat = "pulse";
      e.ph = 'C';
      e.name = name_prefix + ch;
      e.ts_ns = ts;
      e.arg_name = "u";
      e.arg_value = u;

      out_events->push_back(std::move(e));
    }
    return Ok();
  };

  for (Dim i = 0; i < n_points; i += stride) {
    FLUXPULSE_RETURN_IF_ERROR(emit_at(i));
  }

  // Always include last sample if stride doesn't hit it.
  const Dim last = n_points - 1;
  if (last >= 0 && (last % stride) != 0) {
    // Rebuild cursor and advance deterministically to last (simple & robust).
    cursor = program.make_cursor();
    for (Dim i = 0; i <= last; ++i) {
      for (Dim pidx = 0; pidx < static_cast<Dim>(program.pulses.size()); ++pidx) {
        (void)program.coeff_pulse_on_grid(pidx, i, &cursor);
      }
    }
    FLUXPULSE_RETURN_IF_ERROR(emit_at(last));
  }

  return Ok();
}

Status append_run_span_event(const PulseProgram& program,
                            Real time_scale_to_us,
                            const std::string& name,
                            std::vector<TraceEvent>* out_events) {
  if (!out_events) return Error(ErrorCode::kInvalidArgument, "append_run_span_event: out_events is null");
  if (program.grid.size() <= 0) return Error(ErrorCode::kInvalidArgument, "append_run_span_event: empty time grid");

  const Real t0 = program.grid.t(0);
  const Real t1 = program.grid.t(program.grid.size() - 1);

  TraceEvent e;
  e.pid = kDefaultPid;
  e.tid = kSolverTid;
  e.cat = "solver";
  e.ph = 'X';
  e.name = name.empty() ? std::string("mcwf_run") : name;
  e.ts_ns = to_ns(t0, time_scale_to_us);
  e.dur_ns = dur_ns_from(t0, t1, time_scale_to_us);

  // Optional: store number of steps-ish as arg
  e.arg_name = "n_steps";
  e.arg_value = static_cast<Real>(program.grid.size());

  out_events->push_back(std::move(e));
  return Ok();
}

Status append_mcwf_diagnostic_counter_events(Span<const Real> tlist,
                                            Span<const Real> norm2,
                                            Span<const Real> p_total,
                                            Span<const int32_t> jumped,
                                            Span<const int32_t> jump_index,
                                            Real time_scale_to_us,
                                            const std::string& track_name,
                                            int tid,
                                            std::vector<TraceEvent>* out_events) {
  if (!out_events) return Error(ErrorCode::kInvalidArgument, "append_mcwf_diagnostic_counter_events: out_events is null");
  if (tid == 0) tid = kSolverTid;

  const Dim n = tlist.size;
  if (norm2.size != n || p_total.size != n || jumped.size != n || jump_index.size != n) {
    return Error(ErrorCode::kInvalidArgument, "append_mcwf_diagnostic_counter_events: size mismatch");
  }

  for (Dim i = 0; i < n; ++i) {
    TraceEvent e;
    e.pid = kDefaultPid;
    e.tid = tid;
    e.cat = "mcwf";
    e.ph = 'C';
    e.name = track_name.empty() ? std::string("mcwf") : track_name;
    e.ts_ns = to_ns(tlist[i], time_scale_to_us);

    e.args.reserve(4);
    e.args.push_back(TraceArg{"norm2", norm2[i]});
    e.args.push_back(TraceArg{"p_total", p_total[i]});
    e.args.push_back(TraceArg{"jumped", static_cast<Real>(jumped[i])});
    e.args.push_back(TraceArg{"jump_index", static_cast<Real>(jump_index[i])});

    out_events->push_back(std::move(e));
  }

  return Ok();
}


Status append_mcwf_jump_instant_events(Span<const Real> t_jump,
                                      Span<const int32_t> jump_index,
                                      Real time_scale_to_us,
                                      const std::string& track_name,
                                      int tid,
                                      std::vector<TraceEvent>* out_events) {
  if (!out_events) return Error(ErrorCode::kInvalidArgument, "append_mcwf_jump_instant_events: out_events is null");
  if (tid == 0) tid = kJumpsTid;

  const Dim n = t_jump.size;
  if (jump_index.size != n) {
    return Error(ErrorCode::kInvalidArgument, "append_mcwf_jump_instant_events: size mismatch");
  }

  for (Dim i = 0; i < n; ++i) {
    TraceEvent e;
    e.pid = kDefaultPid;
    e.tid = tid;
    e.cat = "mcwf";
    e.ph = 'i';
    e.scope = 't';
    e.name = track_name.empty() ? std::string("jumps") : track_name;
    e.ts_ns = to_ns(t_jump[i], time_scale_to_us);

    e.args.reserve(1);
    e.args.push_back(TraceArg{"jump_index", static_cast<Real>(jump_index[i])});

    out_events->push_back(std::move(e));
  }

  return Ok();
}

}  // namespace io
}  // namespace fluxpulse
