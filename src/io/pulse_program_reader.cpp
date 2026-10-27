// src/io/pulse_program_reader.cpp
#include "fluxpulse/io/pulse_program_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/control/envelope.hpp"
#include "fluxpulse/control/pulse.hpp"
#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/control/segment.hpp"
#include "fluxpulse/control/time_grid.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/io/json_reader.hpp"

namespace fluxpulse {
namespace {

// -----------------------
// String helpers
// -----------------------

inline std::string lower_ascii(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') out.push_back(char(c - 'A' + 'a'));
    else out.push_back(c);
  }
  return out;
}

// -----------------------
// Small JSON helpers (local-only; avoid naming conflicts with json_reader.hpp)
// -----------------------

Status ppr_as_i64(const JsonValue& v, Dim* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "integer out is null");
  if (!v.is_number()) return Error(ErrorCode::kInvalidArgument, "expected integer (number)");
  const double x = v.number_value;
  if (!std::isfinite(x)) return Error(ErrorCode::kInvalidArgument, "integer must be finite");
  const double r = std::round(x);
  if (std::fabs(x - r) > 1e-9) return Error(ErrorCode::kInvalidArgument, "expected integer value");
  if (r < double(std::numeric_limits<Dim>::min()) || r > double(std::numeric_limits<Dim>::max())) {
    return Error(ErrorCode::kInvalidArgument, "integer out of range");
  }
  *out = static_cast<Dim>(r);
  return Ok();
}

Status ppr_get_optional_object(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "out is null");
  const JsonValue* v = obj.find(key);
  if (!v) {
    *out = nullptr;
    return Ok();
  }
  if (!v->is_object()) return Error(ErrorCode::kInvalidArgument, std::string(key) + " must be an object");
  *out = v;
  return Ok();
}

Status ppr_get_optional_number(const JsonValue& obj, std::string_view key, Real* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "out is null");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  return ::fluxpulse::as_number(*v, out);
}

Status ppr_get_optional_string(const JsonValue& obj, std::string_view key, std::string* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "out is null");
  const JsonValue* v = obj.find(key);
  if (!v) return Ok();
  if (!v->is_string()) return Error(ErrorCode::kInvalidArgument, std::string(key) + " must be a string");
  out->assign(v->string_value.data(), v->string_value.size());
  return Ok();
}

Status ppr_get_optional_bool(const JsonValue& obj, std::string_view key, bool* out, bool default_value) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "out is null");
  const JsonValue* v = obj.find(key);
  if (!v) {
    *out = default_value;
    return Ok();
  }
  if (!v->is_bool()) return Error(ErrorCode::kInvalidArgument, std::string(key) + " must be a bool");
  *out = v->bool_value;
  return Ok();
}

Status ppr_get_optional_array(const JsonValue& obj, std::string_view key, const JsonValue** out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "out is null");
  const JsonValue* v = obj.find(key);
  if (!v) {
    *out = nullptr;
    return Ok();
  }
  if (!v->is_array()) return Error(ErrorCode::kInvalidArgument, std::string(key) + " must be an array");
  *out = v;
  return Ok();
}

// -----------------------
// Phase-1 program parsing (same schema as config.program)
// -----------------------

Status parse_channel_id(const JsonValue& v, ChannelId* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_channel_id: out is null");
  *out = ChannelId::Invalid();

  // Numeric channel id.
  if (v.is_number()) {
    Dim x = 0;
    FLUXPULSE_RETURN_IF_ERROR(ppr_as_i64(v, &x));
    if (x < 0 || x > 65535) return Error(ErrorCode::kInvalidArgument, "channel id out of range");
    *out = ChannelId{static_cast<uint16_t>(x)};
    return Ok();
  }

  if (!v.is_string()) return Error(ErrorCode::kInvalidArgument, "channel must be a number or string");

  const std::string s0 = lower_ascii(std::string_view(v.string_value.data(), v.string_value.size()));

  // Canonical single-qubit aliases.
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

  // Multi-qubit pattern: drive_xN / drive_yN / drive_zN (N is integer qubit index).
  std::string s = s0;
  uint16_t axis_base = 0;
  size_t pos = std::string::npos;

  auto try_axis = [&](std::string_view prefix, uint16_t base) -> bool {
    if (s.rfind(prefix, 0) != 0) return false;
    axis_base = base;
    pos = prefix.size();
    return true;
  };

  if (!(try_axis("drive_x", 1) || try_axis("drivex", 1) || try_axis("drive_y", 2) || try_axis("drivey", 2) ||
        try_axis("drive_z", 3) || try_axis("drivez", 3))) {
    return Error(ErrorCode::kInvalidArgument,
                 "Unknown channel string (expected drive_x/drive_y/drive_z, or drive_xN/drive_yN/drive_zN)");
  }

  // Skip separators.
  while (pos < s.size()) {
    const char c = s[pos];
    if (c >= '0' && c <= '9') break;
    if (c == ':' || c == '/' || c == '_' || c == '[' || c == ']' || c == '(' || c == ')' || c == ' ') {
      ++pos;
      continue;
    }
    return Error(ErrorCode::kInvalidArgument, "Channel suffix must be an integer qubit index");
  }
  if (pos >= s.size()) return Error(ErrorCode::kInvalidArgument, "Channel string missing qubit index");

  Dim q = 0;
  {
    Dim acc = 0;
    for (; pos < s.size(); ++pos) {
      const char c = s[pos];
      if (c < '0' || c > '9') break;
      acc = acc * 10 + (c - '0');
      if (acc > 1000000) return Error(ErrorCode::kInvalidArgument, "Qubit index too large");
    }
    q = acc;
  }

  const Dim raw = static_cast<Dim>(axis_base) + static_cast<Dim>(3) * q;
  if (raw < 0 || raw > 65535) return Error(ErrorCode::kInvalidArgument, "computed channel id out of range");
  *out = ChannelId{static_cast<uint16_t>(raw)};
  return Ok();
}

Status parse_time_grid(const JsonValue& grid_obj, TimeGrid* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_time_grid: out is null");
  if (!grid_obj.is_object()) return Error(ErrorCode::kInvalidArgument, "program.grid must be an object");

  const JsonValue* tlist_v = grid_obj.find("tlist");
  if (tlist_v && tlist_v->is_array()) {
    std::vector<Real> tlist;
    tlist.reserve(tlist_v->array_values.size());
    for (const JsonValue& xv : tlist_v->array_values) {
      Real x = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(::fluxpulse::as_number(xv, &x));
      tlist.push_back(x);
    }
    *out = TimeGrid::Explicit(std::move(tlist));
    return out->validate();
  }

  Real t0 = Real{0};
  (void)ppr_get_optional_number(grid_obj, "t0", &t0);

  Real dt = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(get_required_number(grid_obj, "dt", &dt));

  Dim n_points = 0;
  Status st_np = get_required_dim(grid_obj, "n_points", &n_points);
  if (!st_np.ok()) {
    st_np = get_required_dim(grid_obj, "n_steps", &n_points);
    if (!st_np.ok()) return Annotate(st_np, "program.grid.(n_points or n_steps)");
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

  const bool is_const = lower_ascii(type) == "piecewise_constant" || lower_ascii(type) == "constant" ||
                        lower_ascii(type) == "zoh";
  const bool is_lin = lower_ascii(type) == "piecewise_linear" || lower_ascii(type) == "linear";

  if (!is_const && !is_lin) {
    return Error(ErrorCode::kInvalidArgument,
                 "Envelope type must be piecewise_constant (zoh) or piecewise_linear");
  }

  for (const JsonValue& seg_v : segs->array_values) {
    if (!seg_v.is_object()) return Error(ErrorCode::kInvalidArgument, "Envelope segment must be an object");

    Real t0 = Real{0};
    Real t1 = Real{0};
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(seg_v, "t0", &t0));
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(seg_v, "t1", &t1));

    if (is_const) {
      Real value = Real{0};
      Status stv = ppr_get_optional_number(seg_v, "value", &value);
      if (!stv.ok()) {
        stv = get_required_number(seg_v, "v0", &value);
        if (!stv.ok()) return Annotate(stv, "segment.value");
      }
      out->add(Segment::PiecewiseConstant(t0, t1, value));
    } else {
      Real v0 = Real{0};
      Real v1 = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(seg_v, "v0", &v0));
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(seg_v, "v1", &v1));
      out->add(Segment::PiecewiseLinear(t0, t1, v0, v1));
    }
  }

  // Do not call finalize() here; PulseProgram::validate(true) finalizes all envelopes.
  return Ok();
}

Status parse_pulse_program_object(const JsonValue& program_obj, PulseProgram* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_pulse_program_object: out is null");
  if (!program_obj.is_object()) return Error(ErrorCode::kInvalidArgument, "program must be an object");

  const JsonValue* grid_obj = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(get_required_object(program_obj, "grid", &grid_obj));

  TimeGrid grid;
  FLUXPULSE_RETURN_IF_ERROR(parse_time_grid(*grid_obj, &grid));

  PulseProgram prog(grid);

  const JsonValue* pulses_arr = nullptr;
  Status st_pulses = ppr_get_optional_array(program_obj, "pulses", &pulses_arr);
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

  Status st = prog.validate(/*finalize_envelopes=*/true);
  if (!st.ok()) return Annotate(st, "program.validate");

  *out = std::move(prog);
  return Ok();
}

// -----------------------
// Compiled program conversion
// -----------------------

struct ChannelEnv {
  ChannelId ch;
  Envelope env;
};

Envelope* get_or_add_env(std::vector<ChannelEnv>* chans, ChannelId ch) {
  for (auto& ce : *chans) {
    if (ce.ch == ch) return &ce.env;
  }
  chans->push_back(ChannelEnv{ch, Envelope{}});
  return &chans->back().env;
}

struct TimeScale {
  Real to_us = Real{1};  // multiply file-time values by to_us to get microseconds
};

Status parse_time_unit(const JsonValue& root, TimeScale* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_time_unit: out is null");
  out->to_us = Real{1};

  const JsonValue* v = root.find("time_unit");
  if (!v) return Ok();
  if (!v->is_string()) return Error(ErrorCode::kInvalidArgument, "time_unit must be a string");

  std::string u(v->string_value.data(), v->string_value.size());
  u = lower_ascii(u);

  if (u == "us" || u == "microsecond" || u == "microseconds") { out->to_us = Real{1}; return Ok(); }
  if (u == "ns" || u == "nanosecond" || u == "nanoseconds") { out->to_us = Real{1e-3}; return Ok(); }
  if (u == "ms" || u == "millisecond" || u == "milliseconds") { out->to_us = Real{1e3}; return Ok(); }
  if (u == "s"  || u == "sec" || u == "second" || u == "seconds") { out->to_us = Real{1e6}; return Ok(); }

  return Error(ErrorCode::kInvalidArgument, "Unsupported time_unit (expected s/ms/us/ns)");
}

Real round_to_grid(Real t, Real dt, std::string_view rounding) {
  if (dt <= Real{0}) return t;
  const Real x = t / dt;
  if (rounding == "ceil") return dt * std::ceil(x);
  if (rounding == "floor") return dt * std::floor(x);
  return dt * std::round(x);  // nearest
}

struct GridHint {
  Real dt_us = Real{0.001};
  bool align_to_dt = false;
  std::string rounding = "nearest";
};

Status parse_grid_hint(const JsonValue& root, const TimeScale& ts, GridHint* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_grid_hint: out is null");
  GridHint gh;

  const JsonValue* hint = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(ppr_get_optional_object(root, "grid_hint", &hint));
  if (hint) {
    (void)ppr_get_optional_bool(*hint, "align_to_dt", &gh.align_to_dt, false);
    (void)ppr_get_optional_string(*hint, "rounding", &gh.rounding);

    Real dt = Real{0};
    Status st_dt = ppr_get_optional_number(*hint, "dt", &dt);
    if (!st_dt.ok()) return Annotate(st_dt, "grid_hint.dt");
    if (dt > Real{0}) gh.dt_us = dt * ts.to_us;
  }

  *out = gh;
  return Ok();
}

Dim inferred_num_qubits_from_config(const Config& cfg) {
  if (cfg.model_type == "coupled_qubits") return cfg.coupled_qubits.nqubits;
  if (cfg.model_type == "generic_qubit") return 1;
  return 1;
}

Status resolve_drive_channels(const Config& cfg, Dim q, ChannelId* chx, ChannelId* chy, bool* has_y) {
  if (!chx || !chy || !has_y) return Error(ErrorCode::kInvalidArgument, "resolve_drive_channels: null out");

  *has_y = false;
  *chy = ChannelId::Invalid();

  if (cfg.model_type == "coupled_qubits") {
    const auto& p = cfg.coupled_qubits;

    if (!p.enable_drive_x) return Error(ErrorCode::kInvalidArgument, "rotate requested but drive_x disabled");

    // X channel list is optional; if empty use default numbering convention.
    if (!p.drive_x.empty()) {
      if (q < 0 || q >= static_cast<Dim>(p.drive_x.size())) {
        return Error(ErrorCode::kInvalidArgument, "rotate: qubit out of range for drive_x list");
      }
      *chx = p.drive_x[static_cast<size_t>(q)];
    } else {
      const Dim id = 1 + 3 * q;
      if (id < 0 || id > 65535) return Error(ErrorCode::kInvalidArgument, "computed drive_x channel out of range");
      *chx = ChannelId{static_cast<uint16_t>(id)};
    }

    *has_y = p.enable_drive_y;
    if (!p.enable_drive_y) return Ok();

    if (!p.drive_y.empty()) {
      if (q < 0 || q >= static_cast<Dim>(p.drive_y.size())) {
        return Error(ErrorCode::kInvalidArgument, "rotate: qubit out of range for drive_y list");
      }
      *chy = p.drive_y[static_cast<size_t>(q)];
    } else {
      const Dim id = 2 + 3 * q;
      if (id < 0 || id > 65535) return Error(ErrorCode::kInvalidArgument, "computed drive_y channel out of range");
      *chy = ChannelId{static_cast<uint16_t>(id)};
    }
    return Ok();
  }

  if (cfg.model_type == "generic_qubit") {
    *chx = ChannelId::DriveX();
    *has_y = true;
    *chy = ChannelId::DriveY();
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument, "compiled program conversion supports only coupled_qubits/generic_qubit");
}

struct ShapeSpec {
  std::string type = "square";
  Real sigma_us = Real{0};
  Real truncate = Real{3};
};

Status parse_shape_spec(const JsonValue& instr_obj, const TimeScale& ts, ShapeSpec* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "parse_shape_spec: out is null");
  ShapeSpec s;

  const JsonValue* shape = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(ppr_get_optional_object(instr_obj, "shape", &shape));
  if (!shape) {
    *out = s;
    return Ok();
  }

  std::string type;
  FLUXPULSE_RETURN_IF_ERROR(get_required_string(*shape, "type", &type));
  s.type = lower_ascii(type);

  if (s.type == "square" || s.type == "const" || s.type == "constant" || s.type == "zoh") {
    *out = s;
    return Ok();
  }
  if (s.type == "gaussian") {
    Real sigma = Real{0};
    FLUXPULSE_RETURN_IF_ERROR(get_required_number(*shape, "sigma", &sigma));
    s.sigma_us = sigma * ts.to_us;
    (void)ppr_get_optional_number(*shape, "truncate", &s.truncate);
    if (s.sigma_us <= Real{0}) return Error(ErrorCode::kInvalidArgument, "gaussian.sigma must be > 0");
    *out = s;
    return Ok();
  }

  return Error(ErrorCode::kInvalidArgument, "Unsupported shape.type (supported: square, gaussian)");
}

Real gaussian_value(Real t_us, Real T_us, Real sigma_us) {
  const Real center = Real{0.5} * T_us;
  const Real x = (t_us - center) / sigma_us;
  return std::exp(Real{-0.5} * x * x);
}

// Convention used here matches coupled_qubits control terms as implemented in Phase 1:
// - Single-qubit drive Hamiltonian uses 0.5*X and 0.5*Y operators
// - Envelope amplitude integrates to rotation angle: theta = ∫ amp(t) dt
Status emit_rotate_segments(Real t0_us,
                           Real duration_us,
                           Dim qubit,
                           Real axis_phase_rad,
                           Real angle_rad,
                           const ShapeSpec& shape,
                           const Config& cfg,
                           const std::vector<Real>& frame_z,
                           const GridHint& gh,
                           std::vector<ChannelEnv>* out_envs) {
  if (!out_envs) return Error(ErrorCode::kInvalidArgument, "emit_rotate_segments: out_envs is null");
  if (duration_us <= Real{0}) return Error(ErrorCode::kInvalidArgument, "rotate.duration must be > 0");

  ChannelId chx, chy;
  bool has_y = false;
  FLUXPULSE_RETURN_IF_ERROR(resolve_drive_channels(cfg, qubit, &chx, &chy, &has_y));

  const Real phi = axis_phase_rad + frame_z[static_cast<size_t>(qubit)];
  const Real s = std::sin(phi);
  const Real c = std::cos(phi);
  const Real eps = Real{1e-12};

  if (!has_y && std::fabs(s) > eps) {
    return Error(ErrorCode::kInvalidArgument, "rotate requires Y component but drive_y is disabled");
  }

  const Real dt = gh.dt_us;
  if (dt <= Real{0}) return Error(ErrorCode::kInvalidArgument, "internal error: dt must be > 0");

  const Real t1_us = t0_us + duration_us;

  // Build bins sequentially using seg_t0 = previous seg_t1. This guarantees no overlaps even
  // under floating-point rounding and produces contiguous segments within [t0_us, t1_us].
  struct Bin {
    Real t0;
    Real t1;
    Real sval;
  };
  std::vector<Bin> bins;
  bins.reserve(static_cast<size_t>(std::ceil(duration_us / dt) + 2));

  Real area = Real{0};
  Real seg_t0 = t0_us;
  for (;;) {
    if (seg_t0 >= t1_us - Real{1e-15}) break;

    Real seg_t1 = seg_t0 + dt;
    if (seg_t1 > t1_us) seg_t1 = t1_us;

    const Real seg_dt = seg_t1 - seg_t0;
    if (seg_dt <= Real{0}) break;

    Real sval = Real{1};
    if (shape.type == "gaussian") {
      const Real mid = seg_t0 + Real{0.5} * seg_dt;
      const Real local_t = mid - t0_us;  // [0, duration]
      sval = gaussian_value(local_t, duration_us, shape.sigma_us);
    }

    bins.push_back(Bin{seg_t0, seg_t1, sval});
    area += sval * seg_dt;

    seg_t0 = seg_t1;
  }

  if (bins.empty()) return Error(ErrorCode::kInvalidArgument, "rotate produced no samples");
  if (area <= Real{0}) return Error(ErrorCode::kInvalidArgument, "rotate shape area is zero");

  // coupled_qubits convention: theta = ∫ amp(t) dt.
  const Real scale = angle_rad / area;

  Envelope* env_x = get_or_add_env(out_envs, chx);
  Envelope* env_y = has_y ? get_or_add_env(out_envs, chy) : nullptr;

  for (const Bin& b : bins) {
    const Real amp = scale * b.sval;
    const Real ax = amp * c;
    const Real ay = amp * s;

    env_x->add(Segment::PiecewiseConstant(b.t0, b.t1, ax));
    if (env_y && std::fabs(ay) > eps) {
      env_y->add(Segment::PiecewiseConstant(b.t0, b.t1, ay));
    }
  }

  return Ok();
}

Status find_zz_strength(const CoupledQubitsModelConfig& p, Dim q0, Dim q1, Real* out_strength) {
  if (!out_strength) return Error(ErrorCode::kInvalidArgument, "find_zz_strength: out is null");
  for (const auto& c : p.zz) {
    if ((c.q0 == q0 && c.q1 == q1) || (c.q0 == q1 && c.q1 == q0)) {
      *out_strength = c.strength;
      return Ok();
    }
  }
  return Error(ErrorCode::kInvalidArgument, "zz_phase: no ZZ coupling found for the requested qubit pair");
}

Status zz_phase_to_duration_us(const Config& cfg, Dim q0, Dim q1, Real phase_rad, std::string_view frame, Real* out_tau_us) {
  if (!out_tau_us) return Error(ErrorCode::kInvalidArgument, "zz_phase_to_duration_us: out is null");
  if (phase_rad == Real{0}) {
    *out_tau_us = Real{0};
    return Ok();
  }
  if (cfg.model_type != "coupled_qubits") {
    return Error(ErrorCode::kInvalidArgument, "zz_phase currently supported only for model_type=coupled_qubits");
  }

  Real zeta = Real{0};
  FLUXPULSE_RETURN_IF_ERROR(find_zz_strength(cfg.coupled_qubits, q0, q1, &zeta));
  if (zeta == Real{0}) return Error(ErrorCode::kInvalidArgument, "zz_phase: ZZ coupling strength is zero");

  const std::string f = lower_ascii(frame);
  // coupled_qubits convention: Hzz = zeta * (Z⊗Z)
  // frame "ZZ_over_2": target U = exp(-i (phase/2) Z⊗Z) => zeta*tau = phase/2
  if (f.empty() || f == "zz_over_2" || f == "zz/2") {
    *out_tau_us = phase_rad / (Real{2} * zeta);
    return Ok();
  }
  if (f == "zz") {
    *out_tau_us = phase_rad / zeta;
    return Ok();
  }
  return Error(ErrorCode::kInvalidArgument, "zz_phase.frame unsupported (expected ZZ_over_2 or ZZ)");
}

}  // namespace

Status load_pulse_program_json_file(const std::string& path, PulseProgram* out, size_t* error_pos) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "load_pulse_program_json_file: out is null");

  JsonValue root;
  Status st = read_json_file(path, &root, error_pos);
  if (!st.ok()) return Annotate(st, "load_pulse_program_json_file: read_json_file");
  if (!root.is_object()) return Error(ErrorCode::kInvalidArgument, "pulse program JSON root must be an object");

  // Wrapped {"program": {...}} or standalone {"grid":..., "pulses":...}
  const JsonValue* program_obj = root.find("program");
  if (program_obj) {
    if (!program_obj->is_object()) return Error(ErrorCode::kInvalidArgument, "program must be an object");
    return parse_pulse_program_object(*program_obj, out);
  }
  return parse_pulse_program_object(root, out);
}

Status load_compiled_program_json_file(const std::string& path,
                                      const Config& cfg,
                                      PulseProgram* out,
                                      size_t* error_pos) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "load_compiled_program_json_file: out is null");

  JsonValue root;
  Status st = read_json_file(path, &root, error_pos);
  if (!st.ok()) return Annotate(st, "load_compiled_program_json_file: read_json_file");
  if (!root.is_object()) return Error(ErrorCode::kInvalidArgument, "compiled program root must be an object");

  Dim version = 0;
  {
    const JsonValue* v = root.find("version");
    if (!v) return Error(ErrorCode::kInvalidArgument, "compiled program missing required field: version");
    FLUXPULSE_RETURN_IF_ERROR(ppr_as_i64(*v, &version));
    if (version != 1) return Error(ErrorCode::kInvalidArgument, "unsupported compiled program version (expected 1)");
  }

  const Dim nqubits = inferred_num_qubits_from_config(cfg);
  if (nqubits <= 0) return Error(ErrorCode::kInvalidArgument, "invalid number of qubits inferred from config");

  TimeScale ts;
  FLUXPULSE_RETURN_IF_ERROR(parse_time_unit(root, &ts));

  GridHint gh;
  FLUXPULSE_RETURN_IF_ERROR(parse_grid_hint(root, ts, &gh));

  // schedule.type: only asap supported
  std::string sched_type = "asap";
  const JsonValue* sched = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(ppr_get_optional_object(root, "schedule", &sched));
  if (sched) {
    (void)ppr_get_optional_string(*sched, "type", &sched_type);
    sched_type = lower_ascii(sched_type);
    if (sched_type.empty()) sched_type = "asap";
    if (sched_type != "asap") return Error(ErrorCode::kInvalidArgument, "only schedule.type=asap is supported");
  }

  const JsonValue* instrs = nullptr;
  FLUXPULSE_RETURN_IF_ERROR(get_required_array(root, "instructions", &instrs));

  std::vector<Real> frame_z(static_cast<size_t>(nqubits), Real{0});
  std::vector<ChannelEnv> envs;
  envs.reserve(16);

  Real t_cursor = Real{0};

  for (const JsonValue& instr : instrs->array_values) {
    if (!instr.is_object()) return Error(ErrorCode::kInvalidArgument, "instruction must be an object");

    std::string op;
    FLUXPULSE_RETURN_IF_ERROR(get_required_string(instr, "op", &op));
    op = lower_ascii(op);

    if (op == "frame_z") {
      Dim q = 0;
      Real phase = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(get_required_dim(instr, "qubit", &q));
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(instr, "phase", &phase));
      if (q < 0 || q >= nqubits) return Error(ErrorCode::kInvalidArgument, "frame_z.qubit out of range");
      frame_z[static_cast<size_t>(q)] += phase;
      continue;
    }

    if (op == "barrier") {
      std::string align;
      (void)ppr_get_optional_string(instr, "align", &align);
      align = lower_ascii(align);
      if (align == "dt") {
        t_cursor = round_to_grid(t_cursor, gh.dt_us, gh.rounding);
      }
      continue;
    }

    if (op == "wait") {
      Real dur = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(instr, "duration", &dur));
      dur *= ts.to_us;
      if (dur < Real{0}) return Error(ErrorCode::kInvalidArgument, "wait.duration must be >= 0");
      if (gh.align_to_dt) dur = round_to_grid(dur, gh.dt_us, "ceil");
      t_cursor += dur;
      continue;
    }

    if (op == "zz_phase") {
      const JsonValue* qs = nullptr;
      FLUXPULSE_RETURN_IF_ERROR(get_required_array(instr, "qubits", &qs));
      if (qs->array_values.size() != 2) return Error(ErrorCode::kInvalidArgument, "zz_phase.qubits must have length 2");

      Dim q0 = 0, q1 = 0;
      FLUXPULSE_RETURN_IF_ERROR(ppr_as_i64(qs->array_values[0], &q0));
      FLUXPULSE_RETURN_IF_ERROR(ppr_as_i64(qs->array_values[1], &q1));
      if (q0 < 0 || q0 >= nqubits || q1 < 0 || q1 >= nqubits) {
        return Error(ErrorCode::kInvalidArgument, "zz_phase.qubits out of range");
      }

      Real phase = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(instr, "phase", &phase));

      std::string frame = "ZZ_over_2";
      (void)ppr_get_optional_string(instr, "frame", &frame);

      Real tau = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(zz_phase_to_duration_us(cfg, q0, q1, phase, frame, &tau));
      if (tau < Real{0}) return Error(ErrorCode::kInvalidArgument, "zz_phase produced negative duration");

      if (gh.align_to_dt) tau = round_to_grid(tau, gh.dt_us, "ceil");
      t_cursor += tau;
      continue;
    }

    if (op == "rotate") {
      Dim q = 0;
      Real angle = Real{0};
      Real dur = Real{0};

      FLUXPULSE_RETURN_IF_ERROR(get_required_dim(instr, "qubit", &q));
      if (q < 0 || q >= nqubits) return Error(ErrorCode::kInvalidArgument, "rotate.qubit out of range");

      const JsonValue* axis = nullptr;
      FLUXPULSE_RETURN_IF_ERROR(get_required_object(instr, "axis", &axis));

      std::string plane;
      FLUXPULSE_RETURN_IF_ERROR(get_required_string(*axis, "plane", &plane));
      if (lower_ascii(plane) != "xy") return Error(ErrorCode::kInvalidArgument, "rotate.axis.plane must be 'xy'");

      Real phase = Real{0};
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(*axis, "phase", &phase));

      FLUXPULSE_RETURN_IF_ERROR(get_required_number(instr, "angle", &angle));
      FLUXPULSE_RETURN_IF_ERROR(get_required_number(instr, "duration", &dur));
      dur *= ts.to_us;

      if (gh.align_to_dt) dur = round_to_grid(dur, gh.dt_us, "ceil");

      ShapeSpec shape;
      FLUXPULSE_RETURN_IF_ERROR(parse_shape_spec(instr, ts, &shape));

      FLUXPULSE_RETURN_IF_ERROR(
          emit_rotate_segments(t_cursor, dur, q, phase, angle, shape, cfg, frame_z, gh, &envs));
      t_cursor += dur;
      continue;
    }

    return Error(ErrorCode::kInvalidArgument, std::string("unsupported instruction op: ") + op);
  }

  Real dt = gh.dt_us;
  Real t_end = t_cursor;
  if (gh.align_to_dt) t_end = round_to_grid(t_end, dt, "ceil");

  const Dim n_points = static_cast<Dim>(std::floor(t_end / dt)) + 1;
  if (n_points <= 1) return Error(ErrorCode::kInvalidArgument, "compiled program produced empty grid");

  TimeGrid grid = TimeGrid::Uniform(Real{0}, dt, n_points);
  PulseProgram prog(grid);

  for (auto& ce : envs) {
    prog.add_pulse(Pulse(ce.ch, std::move(ce.env)));
  }

  Status stv = prog.validate(/*finalize_envelopes=*/true);
  if (!stv.ok()) return Annotate(stv, "compiled-program program.validate");

  *out = std::move(prog);
  return Ok();
}

}  // namespace fluxpulse
