// src/io/config_loader.cpp
#include "fluxpulse/io/config_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/control/envelope.hpp"
#include "fluxpulse/control/pulse.hpp"
#include "fluxpulse/control/segment.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse {
namespace {

constexpr Real kEps = Real{1e-12};

inline Real abs_real(Real x) { return (x < Real{0}) ? -x : x; }

bool ieq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const unsigned char ca = static_cast<unsigned char>(a[i]);
    const unsigned char cb = static_cast<unsigned char>(b[i]);
    if (std::tolower(ca) != std::tolower(cb)) return false;
  }
  return true;
}

std::string lower_ascii(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

Status get_optional_object(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_object: out is null");
  *out = nullptr;
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  if (!v->is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object member");
  *out = v;
  return Ok();
}

Status get_optional_array(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_array: out is null");
  *out = nullptr;
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  if (!v->is_array()) return Error(ErrorCode::kInvalidArgument, "JSON: expected array member");
  *out = v;
  return Ok();
}

Status get_optional_string(const JsonValue& obj, std::string_view key, std::string* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_string: out is null");
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  if (!v->is_string()) return Error(ErrorCode::kInvalidArgument, "JSON: expected string member");
  *out = v->string_value;
  return Ok();
}

Status get_optional_bool(const JsonValue& obj, std::string_view key, bool* out, bool default_value) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_bool: out is null");
  *out = default_value;
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  if (!v->is_bool()) return Error(ErrorCode::kInvalidArgument, "JSON: expected bool member");
  *out = v->bool_value;
  return Ok();
}

Status get_optional_number(const JsonValue& obj, std::string_view key, Real* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_number: out is null");
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  if (!v->is_number()) return Error(ErrorCode::kInvalidArgument, "JSON: expected number member");
  *out = static_cast<Real>(v->number_value);
  return Ok();
}

Status get_optional_u64(const JsonValue& obj, std::string_view key, uint64_t* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_u64: out is null");
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  return as_u64(*v, out);
}

Status get_optional_dim(const JsonValue& obj, std::string_view key, Dim* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "get_optional_dim: out is null");
  if (!obj.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  return as_dim(*v, out);
}

// -----------------------------------------------------------------------------
// Channel parsing
// -----------------------------------------------------------------------------

static inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

Status parse_channel_id(const JsonValue& v, ChannelId* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_channel_id: out is null");
  *out = ChannelId::Invalid();

  if (v.is_string()) {
    const std::string s0 = lower_ascii(std::string_view(v.string_value.data(), v.string_value.size()));

    // Fast paths (single-qubit convention)
    if (s0 == "drive_x" || s0 == "drivex") {
      *out = ChannelId::DriveX();
      return Ok();
    }
    if (s0 == "drive_y" || s0 == "drivey") {
      *out = ChannelId::DriveY();
      return Ok();
    }
    if (s0 == "drive_z" || s0 == "drivez") {
      *out = ChannelId::DriveZ();
      return Ok();
    }

    // Multi-qubit pattern: drive_{x|y|z} + optional sep + digits
    std::string s = s0;

    uint16_t axis_base = 0;
    size_t pos = std::string::npos;

    auto try_axis = [&](std::string_view prefix, uint16_t base) -> bool {
      if (s.rfind(prefix, 0) != 0) return false;
      axis_base = base;
      pos = prefix.size();
      return true;
    };

    if (!(try_axis("drive_x", 1) || try_axis("drivex", 1) || try_axis("drivey", 2) || try_axis("drive_y", 2) ||
          try_axis("drivez", 3) || try_axis("drive_z", 3))) {
      return Error(ErrorCode::kInvalidArgument,
                   "Unknown channel string (expected drive_x/drive_y/drive_z, or drive_xN/drive_yN/drive_zN)");
    }

    // Skip optional separators
    while (pos < s.size()) {
      const char c = s[pos];
      if (is_digit(c)) break;
      if (c == ':' || c == '/' || c == '_' || c == '[' || c == ']' || c == '(' || c == ')' || c == ' ') {
        ++pos;
        continue;
      }
      return Error(ErrorCode::kInvalidArgument, "Channel suffix must be an integer qubit index");
    }

    // No digits => interpret as qubit 0 (robust fallback)
    if (pos >= s.size()) {
      *out = ChannelId(axis_base);
      return Ok();
    }

    // Parse qubit index
    uint64_t q = 0;
    while (pos < s.size() && is_digit(s[pos])) {
      q = q * 10u + static_cast<uint64_t>(s[pos] - '0');
      if (q > 1000000u) return Error(ErrorCode::kOutOfRange, "Channel qubit index too large");
      ++pos;
    }

    // Allow trailing bracket/paren/space only
    while (pos < s.size()) {
      if (s[pos] == ' ' || s[pos] == ']' || s[pos] == ')') {
        ++pos;
        continue;
      }
      return Error(ErrorCode::kInvalidArgument, "Unexpected trailing characters in channel string");
    }

    const uint64_t id = static_cast<uint64_t>(axis_base) + 3u * q;
    if (id == 0 || id > 65535u) {
      return Error(ErrorCode::kOutOfRange, "ChannelId out of range after applying multi-qubit mapping");
    }
    *out = ChannelId(static_cast<uint16_t>(id));
    return Ok();
  }

  if (v.is_number()) {
    uint64_t id = 0;
    FLUXPULSE_RETURN_IF_ERROR(as_u64(v, &id));
    if (id == 0 || id > 65535u) {
      return Error(ErrorCode::kOutOfRange, "ChannelId numeric value out of range (1..65535)");
    }
    *out = ChannelId(static_cast<uint16_t>(id));
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument, "ChannelId must be a string or a number");
}

// -----------------------------------------------------------------------------
// Pulse program parsing
// -----------------------------------------------------------------------------

Status parse_time_grid(const JsonValue& grid_obj, TimeGrid* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_time_grid: out is null");
  if (!grid_obj.is_object()) return Error(ErrorCode::kInvalidArgument, "program.grid must be an object");

  // If "tlist" exists, prefer explicit grid.
  const JsonValue* tlist_v = grid_obj.find("tlist");
  if (tlist_v && tlist_v->is_array()) {
    std::vector<Real> tlist;
    tlist.reserve(tlist_v->array_values.size());
    for (const JsonValue& xv : tlist_v->array_values) {
      Real x = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(as_number(xv, &x));
      tlist.push_back(x);
    }
    *out = TimeGrid::Explicit(std::move(tlist));
    return out->validate();
  }

  // Uniform grid.
  Real t0 = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(get_optional_number(grid_obj, "t0", &t0));

  Real dt = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(get_required_number(grid_obj, "dt", &dt));

  Dim n_points = 0;
  Status st_np = get_dim(grid_obj, "n_points", &n_points);
  if (!st_np.ok()) {
    // Allow "n_steps" as alias.
    st_np = get_dim(grid_obj, "n_steps", &n_points);
    if (!st_np.ok()) {
      return Annotate(st_np, "program.grid.(n_points or n_steps)");
    }
  }

  *out = TimeGrid::Uniform(t0, dt, n_points);
  return out->validate();
}

Status parse_envelope(const JsonValue& env_obj, Envelope* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_envelope: out is null");
  if (!env_obj.is_object()) return Error(ErrorCode::kInvalidArgument, "envelope must be an object");

  std::string type;
  FLUXPULSE_RETURN_IF_ERROR(get_required_string(env_obj, "type", &type));

  const JsonValue* segs = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(get_required_array(env_obj, "segments", &segs));

  out->clear();

  const bool is_const = ieq(type, "piecewise_constant") || ieq(type, "constant") || ieq(type, "zoh");
  const bool is_lin = ieq(type, "piecewise_linear") || ieq(type, "linear");

  if (!is_const && !is_lin) {
    return Error(ErrorCode::kInvalidArgument,
                 "Envelope type must be piecewise_constant (zoh) or piecewise_linear");
  }

  for (const JsonValue& seg_v : segs->array_values) {
    if (!seg_v.is_object()) {
      return Error(ErrorCode::kInvalidArgument, "Envelope segment must be an object");
    }

    Real t0 = Real{0};
    Real t1 = Real{0};
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(seg_v, "t0", &t0));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(seg_v, "t1", &t1));

    if (is_const) {
      Real value = Real{0};
      // Accept "value" (preferred) and "v0" as alias.
      Status stv = get_number(seg_v, "value", &value);
      if (!stv.ok()) {
        stv = get_number(seg_v, "v0", &value);
        if (!stv.ok()) return Annotate(stv, "segment.value");
      }
      out->add(Segment::PiecewiseConstant(t0, t1, value));
    } else {
      Real v0 = Real{0};
      Real v1 = Real{0};
      // Preferred keys: v0, v1. Accept value0/value1 aliases.
      Status st0 = get_number(seg_v, "v0", &v0);
      if (!st0.ok()) st0 = get_number(seg_v, "value0", &v0);
      if (!st0.ok()) return Annotate(st0, "segment.v0");

      Status st1 = get_number(seg_v, "v1", &v1);
      if (!st1.ok()) st1 = get_number(seg_v, "value1", &v1);
      if (!st1.ok()) return Annotate(st1, "segment.v1");

      out->add(Segment::PiecewiseLinear(t0, t1, v0, v1));
    }
  }

  // Do not call finalize() here; PulseProgram::validate(true) will finalize all envelopes.
  return Ok();
}

Status parse_pulse_program(const JsonValue& root, PulseProgram* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_pulse_program: out is null");
  if (!root.is_object()) return Error(ErrorCode::kInvalidArgument, "Config root must be an object");

  const JsonValue* program_obj = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(get_required_object(root, "program", &program_obj));

  // Grid
  const JsonValue* grid_obj = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(get_required_object(*program_obj, "grid", &grid_obj));

  TimeGrid grid;
  FLUXPULSE_RETURN_IF_ERROR(parse_time_grid(*grid_obj, &grid));

  PulseProgram prog(grid);

  // Pulses
  const JsonValue* pulses_arr = nullptr;
  Status st_pulses = get_optional_array(*program_obj, "pulses", &pulses_arr);
  if (!st_pulses.ok()) return Annotate(st_pulses, "program.pulses");

  if (pulses_arr) {
    for (const JsonValue& p : pulses_arr->array_values) {
      if (!p.is_object()) return Error(ErrorCode::kInvalidArgument, "Pulse must be an object");

      const JsonValue* ch_v = p.find("channel");
      if (!ch_v) return Error(ErrorCode::kInvalidArgument, "Pulse missing required field: channel");

      ChannelId ch;
      FLUXPULSE_RETURN_IF_ERROR(parse_channel_id(*ch_v, &ch));

      const JsonValue* env_obj = nullptr;
      FLUXPULSE_RETURN_IF_ERROR(get_required_object(p, "envelope", &env_obj));

      Envelope env;
      FLUXPULSE_RETURN_IF_ERROR(parse_envelope(*env_obj, &env));

      prog.add_pulse(Pulse(ch, std::move(env)));
    }
  }

  // Final validate + finalize envelopes (Phase 1 invariant: one pulse per channel).
  Status st = prog.validate(/*finalize_envelopes=*/true);
  if (!st.ok()) return Annotate(st, "program.validate");

  *out = std::move(prog);
  return Ok();
}

// -----------------------------------------------------------------------------
// Run/output/trace parsing
// -----------------------------------------------------------------------------

Status parse_run(const JsonValue& root, const TimeGrid& grid, RunOptionsConfig* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_run: out is null");

  // Defaults
  RunOptionsConfig run;

  const JsonValue* run_obj = nullptr;
  Status st = get_optional_object(root, "run", &run_obj);
  if (!st.ok()) return Annotate(st, "run");

  if (run_obj) {
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*run_obj, "dt", &run.dt));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*run_obj, "t_final", &run.t_final));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_u64(*run_obj, "trajectories", &run.trajectories));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_u64(*run_obj, "base_seed", &run.base_seed));
  }

  // Fill missing values from grid
  if (!(run.dt > Real{0})) {
    if (grid.is_uniform()) {
      run.dt = grid.dt;
    } else if (grid.size() >= 2) {
      run.dt = grid.dt_interval(0);
    }
  }
  if (!(run.t_final > Real{0})) {
    if (grid.size() > 0) {
      run.t_final = grid.t_end();
    }
  }
  if (run.trajectories == 0) {
    run.trajectories = 1;
  }

  // Consistency checks (helpful for debugging)
  if (grid.is_uniform() && (run.dt > Real{0}) && (abs_real(run.dt - grid.dt) > kEps)) {
    return Error(ErrorCode::kInvalidArgument, "run.dt does not match program.grid.dt");
  }
  if (grid.size() > 0 && (run.t_final > Real{0}) && (abs_real(run.t_final - grid.t_end()) > kEps)) {
    return Error(ErrorCode::kInvalidArgument, "run.t_final does not match program.grid end time");
  }

  if (!(run.dt > Real{0})) return Error(ErrorCode::kInvalidArgument, "run.dt must be > 0");
  if (!(run.t_final > Real{0})) return Error(ErrorCode::kInvalidArgument, "run.t_final must be > 0");
  if (run.trajectories == 0) return Error(ErrorCode::kInvalidArgument, "run.trajectories must be > 0");

  *out = run;
  return Ok();
}

Status parse_output_and_trace(const JsonValue& root, OutputConfig* out_output, TraceConfig* out_trace) {
  if (!out_output || !out_trace) {
    return Error(ErrorCode::kInvalidArgument, "parse_output_and_trace: null out ptr");
  }

  OutputConfig output;
  TraceConfig trace;

  // output (optional)
  const JsonValue* out_obj = nullptr;
  Status st = get_optional_object(root, "output", &out_obj);
  if (!st.ok()) return Annotate(st, "output");

  if (out_obj) {
    FLUXPULSE_RETURN_IF_ERROR(get_optional_string(*out_obj, "result_path", &output.result_path));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_string(*out_obj, "trace_path", &output.trace_path));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_bool(*out_obj, "write_trace", &output.write_trace, false));
  }

  // trace (optional)
  const JsonValue* trace_obj = nullptr;
  st = get_optional_object(root, "trace", &trace_obj);
  if (!st.ok()) return Annotate(st, "trace");

  if (trace_obj) {
    FLUXPULSE_RETURN_IF_ERROR(get_optional_bool(*trace_obj, "emit_segments", &trace.emit_segments, true));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_bool(*trace_obj, "emit_counters", &trace.emit_counters, false));

    // Backward-compat: legacy "counter_stride" applies to both pulse counters and diagnostics,
    // unless the new fields are explicitly provided.
    FLUXPULSE_RETURN_IF_ERROR(get_optional_dim(*trace_obj, "counter_stride", &trace.counter_stride));
    if (trace.counter_stride <= 0) trace.counter_stride = 1;

    // Defaults inherit the legacy stride.
    trace.pulse_counter_stride = trace.counter_stride;
    trace.diagnostic_stride = trace.counter_stride;

    // New fields (optional).
    FLUXPULSE_RETURN_IF_ERROR(get_optional_dim(*trace_obj, "pulse_counter_stride", &trace.pulse_counter_stride));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_dim(*trace_obj, "diagnostic_stride", &trace.diagnostic_stride));

    if (trace.pulse_counter_stride <= 0) trace.pulse_counter_stride = 1;
    if (trace.diagnostic_stride <= 0) trace.diagnostic_stride = 1;

    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*trace_obj, "time_scale_to_us", &trace.time_scale_to_us));
    if (trace.time_scale_to_us <= Real{0}) trace.time_scale_to_us = Real{1};
  }

  *out_output = std::move(output);
  *out_trace = std::move(trace);
  return Ok();
}

// -----------------------------------------------------------------------------
// Observables parsing
// -----------------------------------------------------------------------------

Status parse_observables(const JsonValue& root, ObservableConfig* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_observables: out is null");
  out->names.clear();

  const JsonValue* obs_v = root.find("observables");
  if (!obs_v) {
    // Optional; leave empty.
    return Ok();
  }

  if (obs_v->is_array()) {
    for (const JsonValue& x : obs_v->array_values) {
      std::string s;
      FLUXPULSE_RETURN_IF_ERROR(as_string(x, &s));
      out->names.push_back(std::move(s));
    }
    return Ok();
  }

  if (obs_v->is_object()) {
    const JsonValue* names_arr = nullptr;
    FLUXPULSE_RETURN_IF_ERROR(get_required_array(*obs_v, "names", &names_arr));
    for (const JsonValue& x : names_arr->array_values) {
      std::string s;
      FLUXPULSE_RETURN_IF_ERROR(as_string(x, &s));
      out->names.push_back(std::move(s));
    }
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument, "observables must be an array of strings or an object {names:[...]}");
}

// -----------------------------------------------------------------------------
// Helpers for coupled_qubits parsing
// -----------------------------------------------------------------------------

Status parse_real_array_or_broadcast(const JsonValue& parent,
                                    std::string_view key,
                                    Dim n,
                                    std::vector<Real>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_real_array_or_broadcast: out is null");
  out->clear();

  if (!parent.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");

  const JsonValue* v = parent.find(key);
  if (!v) return Ok();

  if (v->is_number()) {
    Real x = Real{0};
    FLUXPULSE_RETURN_IF_ERROR(as_number(*v, &x));
    out->assign(static_cast<size_t>(n), x);
    return Ok();
  }

  if (v->is_array()) {
    if (v->array_values.size() == 1 && n > 1) {
      Real x = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(as_number(v->array_values[0], &x));
      out->assign(static_cast<size_t>(n), x);
      return Ok();
    }
    if (static_cast<Dim>(v->array_values.size()) != n) {
      return Error(ErrorCode::kInvalidArgument, "Array length must equal nqubits (or be scalar / length-1 broadcast)");
    }
    out->reserve(v->array_values.size());
    for (const JsonValue& xv : v->array_values) {
      Real x = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(as_number(xv, &x));
      out->push_back(x);
    }
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument, "Expected number or array");
}

Status parse_channel_array(const JsonValue& parent,
                           std::string_view key,
                           Dim n,
                           std::vector<ChannelId>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_channel_array: out is null");
  out->clear();

  if (!parent.is_object()) return Error(ErrorCode::kInvalidArgument, "JSON: expected object");

  const JsonValue* v = parent.find(key);
  if (!v) return Ok();

  if (!v->is_array()) {
    return Error(ErrorCode::kInvalidArgument, "Expected array of channels");
  }
  if (static_cast<Dim>(v->array_values.size()) != n) {
    return Error(ErrorCode::kInvalidArgument, "Channel array length must equal nqubits");
  }

  out->reserve(v->array_values.size());
  for (const JsonValue& cv : v->array_values) {
    ChannelId ch;
    FLUXPULSE_RETURN_IF_ERROR(parse_channel_id(cv, &ch));
    out->push_back(ch);
  }
  return Ok();
}

Status parse_pair_couplings_array(const JsonValue& arr,
                                  Dim nqubits,
                                  std::vector<PairCouplingConfig>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_pair_couplings_array: out is null");
  out->clear();
  if (!arr.is_array()) return Error(ErrorCode::kInvalidArgument, "Expected array of coupling objects");

  out->reserve(arr.array_values.size());
  for (const JsonValue& cv : arr.array_values) {
    if (!cv.is_object()) return Error(ErrorCode::kInvalidArgument, "Coupling entry must be an object");
    PairCouplingConfig c;

    Dim q0 = 0;
    Dim q1 = 0;
    Real strength = Real{0};

    FLUXPULSE_RETURN_IF_ERROR(get_required_dim(cv, "q0", &q0));
    FLUXPULSE_RETURN_IF_ERROR(get_required_dim(cv, "q1", &q1));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(cv, "strength", &strength));

    if (q0 < 0 || q1 < 0) return Error(ErrorCode::kInvalidArgument, "Coupling indices must be >= 0");
    if (q0 == q1) return Error(ErrorCode::kInvalidArgument, "Coupling requires q0 != q1");
    if (q0 >= nqubits || q1 >= nqubits) return Error(ErrorCode::kInvalidArgument, "Coupling index out of range");

    c.q0 = q0;
    c.q1 = q1;
    c.strength = strength;
    out->push_back(c);
  }
  return Ok();
}

// -----------------------------------------------------------------------------
// Model parsing
// -----------------------------------------------------------------------------

Status parse_model(const JsonValue& root, Config* cfg) {
  if (!cfg) return Error(ErrorCode::kInvalidArgument, "parse_model: cfg is null");

  const JsonValue* model_obj = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(get_required_object(root, "model", &model_obj));

  std::string type;
  FLUXPULSE_RETURN_IF_ERROR(get_required_string(*model_obj, "type", &type));
  cfg->model_type = type;

  // Reset all model blocks to defaults (only one will be populated).
  cfg->generic_qubit = GenericQubitModelConfig{};
  cfg->coupled_qubits = CoupledQubitsModelConfig{};
  cfg->oscillator = OscillatorModelConfig{};
  cfg->kerr_oscillator = KerrOscillatorModelConfig{};
  cfg->duffing_transmon = DuffingTransmonModelConfig{};

  if (ieq(type, "generic_qubit") || ieq(type, "qubit")) {
    GenericQubitModelConfig m;
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*model_obj, "omega", &m.omega));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "detuning", &m.detuning));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "t1", &m.t1));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "tphi", &m.tphi));
    cfg->generic_qubit = m;
    return Ok();
  }

  if (ieq(type, "coupled_qubits") || ieq(type, "coupled") || ieq(type, "2q") || ieq(type, "multi_qubit")) {
    CoupledQubitsModelConfig m;

    FLUXPULSE_RETURN_IF_ERROR(get_required_dim(*model_obj, "nqubits", &m.nqubits));
    if (m.nqubits < 1) return Error(ErrorCode::kInvalidArgument, "coupled_qubits.nqubits must be >= 1");

    // Drift arrays (optional; allow scalar broadcast)
    {
      Status st = parse_real_array_or_broadcast(*model_obj, "omega", m.nqubits, &m.omega);
      if (!st.ok()) return Annotate(st, "model.omega");
    }
    {
      Status st = parse_real_array_or_broadcast(*model_obj, "detuning", m.nqubits, &m.detuning);
      if (!st.ok()) return Annotate(st, "model.detuning");
    }

    // Noise arrays (optional; allow scalar broadcast)
    {
      Status st = parse_real_array_or_broadcast(*model_obj, "t1", m.nqubits, &m.t1);
      if (!st.ok()) return Annotate(st, "model.t1");
    }
    {
      Status st = parse_real_array_or_broadcast(*model_obj, "tphi", m.nqubits, &m.tphi);
      if (!st.ok()) return Annotate(st, "model.tphi");
    }

    // Control flags
    FLUXPULSE_RETURN_IF_ERROR(get_optional_bool(*model_obj, "enable_drive_x", &m.enable_drive_x, true));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_bool(*model_obj, "enable_drive_y", &m.enable_drive_y, false));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_bool(*model_obj, "enable_drive_z", &m.enable_drive_z, false));

    // Optional explicit channel arrays
    {
      Status st = parse_channel_array(*model_obj, "drive_x", m.nqubits, &m.drive_x);
      if (!st.ok()) return Annotate(st, "model.drive_x");
    }
    {
      Status st = parse_channel_array(*model_obj, "drive_y", m.nqubits, &m.drive_y);
      if (!st.ok()) return Annotate(st, "model.drive_y");
    }
    {
      Status st = parse_channel_array(*model_obj, "drive_z", m.nqubits, &m.drive_z);
      if (!st.ok()) return Annotate(st, "model.drive_z");
    }

    // Couplings may be on model object directly, or nested under "couplings".
    const JsonValue* couplings_obj = nullptr;
    FLUXPULSE_RETURN_IF_ERROR(get_optional_object(*model_obj, "couplings", &couplings_obj));

    auto parse_coupling_key =
        [&](const JsonValue& parent, std::string_view key, std::vector<PairCouplingConfig>* out) -> Status {
      const JsonValue* arr = nullptr;
      Status st = get_optional_array(parent, key, &arr);
      if (!st.ok()) return st;
      if (!arr) return Ok();
      return parse_pair_couplings_array(*arr, m.nqubits, out);
    };

    // exchange_xy
    {
      Status st = parse_coupling_key(*model_obj, "exchange_xy", &m.exchange_xy);
      if (!st.ok()) return Annotate(st, "model.exchange_xy");
      if (couplings_obj) {
        std::vector<PairCouplingConfig> tmp;
        Status st2 = parse_coupling_key(*couplings_obj, "exchange_xy", &tmp);
        if (!st2.ok()) return Annotate(st2, "model.couplings.exchange_xy");
        if (!tmp.empty()) m.exchange_xy = std::move(tmp);
      }
    }

    // zz
    {
      Status st = parse_coupling_key(*model_obj, "zz", &m.zz);
      if (!st.ok()) return Annotate(st, "model.zz");
      if (couplings_obj) {
        std::vector<PairCouplingConfig> tmp;
        Status st2 = parse_coupling_key(*couplings_obj, "zz", &tmp);
        if (!st2.ok()) return Annotate(st2, "model.couplings.zz");
        if (!tmp.empty()) m.zz = std::move(tmp);
      }
    }

    cfg->coupled_qubits = std::move(m);
    return Ok();
  }

  if (ieq(type, "oscillator")) {
    OscillatorModelConfig m;
    FLUXPULSE_RETURN_IF_ERROR(get_required_dim(*model_obj, "levels", &m.levels));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*model_obj, "omega", &m.omega));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "tkappa", &m.tkappa));
    if (m.levels < 2) return Error(ErrorCode::kInvalidArgument, "oscillator.levels must be >= 2");
    cfg->oscillator = m;
    return Ok();
  }

  if (ieq(type, "kerr_oscillator") || ieq(type, "kerr")) {
    KerrOscillatorModelConfig m;
    FLUXPULSE_RETURN_IF_ERROR(get_required_dim(*model_obj, "levels", &m.levels));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*model_obj, "omega", &m.omega));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*model_obj, "kerr", &m.kerr));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "tkappa", &m.tkappa));
    if (m.levels < 2) return Error(ErrorCode::kInvalidArgument, "kerr_oscillator.levels must be >= 2");
    cfg->kerr_oscillator = m;
    return Ok();
  }

  if (ieq(type, "duffing_transmon") || ieq(type, "transmon")) {
    DuffingTransmonModelConfig m;
    FLUXPULSE_RETURN_IF_ERROR(get_required_dim(*model_obj, "levels", &m.levels));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*model_obj, "omega", &m.omega));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*model_obj, "anharm", &m.anharm));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "drive_scale", &m.drive_scale));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "t1", &m.t1));
    FLUXPULSE_RETURN_IF_ERROR(get_optional_number(*model_obj, "tphi", &m.tphi));
    if (m.levels < 2) return Error(ErrorCode::kInvalidArgument, "duffing_transmon.levels must be >= 2");
    cfg->duffing_transmon = m;
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument,
               "Unknown model.type (expected generic_qubit, coupled_qubits, oscillator, kerr_oscillator, duffing_transmon)");
}

// -----------------------------------------------------------------------------
// Initial state parsing
// -----------------------------------------------------------------------------

void infer_default_dims_from_model(const Config& cfg, std::vector<Dim>* dims_out) {
  dims_out->clear();
  const std::string t = lower_ascii(cfg.model_type);

  if (t == "coupled_qubits" || t == "coupled" || t == "2q" || t == "multi_qubit") {
    const Dim n = cfg.coupled_qubits.nqubits;
    dims_out->assign(static_cast<size_t>(n), Dim{2});
    return;
  }
  if (t == "generic_qubit" || t == "qubit") {
    dims_out->assign(1, Dim{2});
    return;
  }
  if (t == "oscillator") {
    dims_out->assign(1, cfg.oscillator.levels);
    return;
  }
  if (t == "kerr_oscillator" || t == "kerr") {
    dims_out->assign(1, cfg.kerr_oscillator.levels);
    return;
  }
  if (t == "duffing_transmon" || t == "transmon") {
    dims_out->assign(1, cfg.duffing_transmon.levels);
    return;
  }
  // Unknown => leave empty.
}

Status parse_dims_array(const JsonValue& v, std::vector<Dim>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_dims_array: out is null");
  out->clear();
  if (!v.is_array()) return Error(ErrorCode::kInvalidArgument, "dims must be an array");
  out->reserve(v.array_values.size());
  for (const JsonValue& x : v.array_values) {
    Dim d = 0;
    FLUXPULSE_RETURN_IF_ERROR(as_dim(x, &d));
    if (d < 2) return Error(ErrorCode::kInvalidArgument, "dims entries must be >= 2");
    out->push_back(d);
  }
  return Ok();
}

Status parse_index_digits(const JsonValue& v, std::vector<Dim>* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_index_digits: out is null");
  out->clear();
  if (!v.is_array()) return Error(ErrorCode::kInvalidArgument, "index must be an array for digit form");
  out->reserve(v.array_values.size());
  for (const JsonValue& x : v.array_values) {
    Dim d = 0;
    FLUXPULSE_RETURN_IF_ERROR(as_dim(x, &d));
    if (d < 0) return Error(ErrorCode::kInvalidArgument, "index digits must be >= 0");
    out->push_back(d);
  }
  return Ok();
}

Status validate_basis_index(const std::vector<Dim>& dims, const InitialStateConfig& st) {
  if (!st.has_linear_index) {
    if (!st.index.empty() && st.index.size() != dims.size()) {
      return Error(ErrorCode::kInvalidArgument, "initial_state.index length must match dims length");
    }
    for (size_t i = 0; i < st.index.size(); ++i) {
      if (st.index[i] < 0 || st.index[i] >= dims[i]) {
        return Error(ErrorCode::kInvalidArgument, "initial_state.index digit out of range for dims");
      }
    }
    return Ok();
  }

  // linear_index must be < product(dims)
  uint64_t total = 1;
  for (Dim d : dims) {
    if (d <= 0) return Error(ErrorCode::kInvalidArgument, "initial_state.dims invalid");
    if (total > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(d)) {
      return Error(ErrorCode::kOutOfRange, "initial_state.dims product overflow");
    }
    total *= static_cast<uint64_t>(d);
  }
  if (st.linear_index >= total) {
    return Error(ErrorCode::kOutOfRange, "initial_state.linear_index out of range for dims product");
  }
  return Ok();
}

Status parse_initial_state(const JsonValue& root, const Config& cfg, InitialStateConfig* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_initial_state: out is null");
  *out = InitialStateConfig{};  // defaults to basis

  const JsonValue* v = root.find("initial_state");
  if (!v) {
    // Optional; default to all-zero basis state with inferred dims.
    infer_default_dims_from_model(cfg, &out->dims);
    out->index.assign(out->dims.size(), Dim{0});
    out->has_linear_index = false;
    out->linear_index = 0;
    return Ok();
  }

  if (v->is_string()) {
    // Shorthand forms.
    const std::string s = lower_ascii(std::string_view(v->string_value.data(), v->string_value.size()));
    infer_default_dims_from_model(cfg, &out->dims);
    out->type = "basis";
    out->has_linear_index = false;
    out->linear_index = 0;
    out->index.assign(out->dims.size(), Dim{0});

    if (s == "ground" || s == "zero" || s == "|0>" || s == "|00>" || s == "basis0") {
      return Ok();
    }
    if (s == "excited" || s == "one" || s == "|1>") {
      if (!out->index.empty()) out->index[0] = 1;
      return Ok();
    }
    return Error(ErrorCode::kInvalidArgument, "initial_state string must be ground/zero/excited/one");
  }

  if (!v->is_object()) {
    return Error(ErrorCode::kInvalidArgument, "initial_state must be an object or a shorthand string");
  }

  std::string type;
  FLUXPULSE_RETURN_IF_ERROR(get_required_string(*v, "type", &type));
  out->type = type;

  if (!(ieq(type, "basis"))) {
    return Error(ErrorCode::kInvalidArgument, "initial_state.type: Phase 1 supports only 'basis'");
  }

  // dims (optional)
  const JsonValue* dims_v = v->find("dims");
  if (dims_v) {
    FLUXPULSE_RETURN_IF_ERROR(parse_dims_array(*dims_v, &out->dims));
  } else {
    infer_default_dims_from_model(cfg, &out->dims);
  }

  // index (optional)
  const JsonValue* idx_v = v->find("index");
  if (idx_v) {
    if (idx_v->is_array()) {
      FLUXPULSE_RETURN_IF_ERROR(parse_index_digits(*idx_v, &out->index));
      out->has_linear_index = false;
      out->linear_index = 0;
    } else if (idx_v->is_number()) {
      uint64_t li = 0;
      FLUXPULSE_RETURN_IF_ERROR(as_u64(*idx_v, &li));
      out->has_linear_index = true;
      out->linear_index = li;
      out->index.clear();
    } else {
      return Error(ErrorCode::kInvalidArgument, "initial_state.index must be an array of ints or an integer");
    }
  } else {
    // default all-zero
    out->has_linear_index = false;
    out->linear_index = 0;
    out->index.assign(out->dims.size(), Dim{0});
  }

  // Validate
  FLUXPULSE_RETURN_IF_ERROR(validate_basis_index(out->dims, *out));
  return Ok();
}

}  // namespace

Status load_config_file(const std::string& path, Config* cfg, size_t* error_pos) {
  if (!cfg) {
    if (error_pos) *error_pos = 0;
    return Error(ErrorCode::kInvalidArgument, "load_config_file: cfg is null");
  }

  JsonValue root;
  Status st = read_json_file(path, &root, error_pos);
  if (!st.ok()) return Annotate(st, "load_config_file: read_json_file");

  st = load_config_from_json(root, cfg);
  if (!st.ok()) return Annotate(st, "load_config_file: load_config_from_json");

  return Ok();
}

Status load_config_from_json(const JsonValue& root, Config* cfg) {
  if (!cfg) return Error(ErrorCode::kInvalidArgument, "load_config_from_json: cfg is null");
  if (!root.is_object()) return Error(ErrorCode::kInvalidArgument, "Config root must be a JSON object");

  // Start from defaults.
  *cfg = Config{};

  // Program first (so run defaults can be derived from grid).
  Status st = parse_pulse_program(root, &cfg->program);
  if (!st.ok()) return Annotate(st, "program");

  // Run
  st = parse_run(root, cfg->program.grid, &cfg->run);
  if (!st.ok()) return Annotate(st, "run");

  // Output + trace options
  st = parse_output_and_trace(root, &cfg->output, &cfg->trace);
  if (!st.ok()) return Annotate(st, "output/trace");

  // Model
  st = parse_model(root, cfg);
  if (!st.ok()) return Annotate(st, "model");

  // Initial state (may depend on model dims)
  st = parse_initial_state(root, *cfg, &cfg->initial_state);
  if (!st.ok()) return Annotate(st, "initial_state");

  // Observables
  st = parse_observables(root, &cfg->observables);
  if (!st.ok()) return Annotate(st, "observables");

  return Ok();
}

}  // namespace fluxpulse
