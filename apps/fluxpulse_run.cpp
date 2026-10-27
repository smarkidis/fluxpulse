// apps/fluxpulse_run.cpp
#include <cstdint>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/device/processor.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/io/config_loader.hpp"
#include "fluxpulse/io/result_writer.hpp"
#include "fluxpulse/io/trace_writer.hpp"
#include "fluxpulse/io/pulse_program_reader.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"
#include "fluxpulse/model/effective/duffing_transmon.hpp"
#include "fluxpulse/model/effective/generic_qubit.hpp"
#include "fluxpulse/sim/mcwf/mcwf_solver.hpp"
#include "fluxpulse/sim/observable_builders.hpp"

namespace fluxpulse {
namespace {

inline bool ends_with(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) return false;
  return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lower_ascii(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

inline std::string fmt_seconds(double s) {
  std::ostringstream ss;
  if (s < 60.0) {
    ss << std::fixed << std::setprecision(1) << s << "s";
  } else {
    const int m = static_cast<int>(s / 60.0);
    const double r = s - 60.0 * m;
    ss << m << "m" << std::fixed << std::setprecision(0) << r << "s";
  }
  return ss.str();
}

struct ScopedStage {
  const char* name;
  std::chrono::steady_clock::time_point t0;
  explicit ScopedStage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {
    std::cout << "[fluxpulse_run] " << name << "...\n";
  }
  ~ScopedStage() {
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[fluxpulse_run] " << name << " done (" << fmt_seconds(s) << ")\n";
  }
};

inline Dim compute_num_samples(Dim n_points, Dim stride) {
  if (n_points <= 0) return 0;
  if (stride <= 0) stride = 1;
  const Dim last = n_points - 1;
  Dim n = last / stride + 1;
  if ((last % stride) != 0) n += 1;
  return n;
}

inline void print_progress(uint64_t done,
                           uint64_t total,
                           const std::chrono::steady_clock::time_point& start,
                           bool final) {
  const double frac = (total == 0) ? 1.0 : static_cast<double>(done) / static_cast<double>(total);
  const int width = 30;
  const int filled = static_cast<int>(frac * width + 0.5);
  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - start).count();

  std::ostringstream ss;
  ss << "\r[fluxpulse_run] trajectories: " << done << "/" << total << "  [";
  for (int i = 0; i < width; ++i) ss << (i < filled ? '=' : ' ');
  ss << "] " << std::fixed << std::setprecision(1) << (100.0 * frac) << "%  elapsed " << fmt_seconds(elapsed);
  if (final) ss << "\n";
  std::cout << ss.str() << std::flush;
}



inline void print_list_inline(const std::string& label, const std::vector<ChannelId>& v, size_t max_items = 10) {
  std::cout << "[fluxpulse_run] " << label << " (n=" << v.size() << "): ";
  const size_t n = v.size();
  const size_t m = (n < max_items) ? n : max_items;
  std::cout << "[";
  for (size_t i = 0; i < m; ++i) {
    if (i) std::cout << ", ";
    std::cout << static_cast<unsigned>(v[i].value);
  }
  if (n > m) std::cout << ", ...";
  std::cout << "]\n";
}

template <typename T>
inline void print_list_inline(const std::string& label, const std::vector<T>& v, size_t max_items = 10) {
  std::cout << "[fluxpulse_run] " << label << " (n=" << v.size() << "): ";
  const size_t n = v.size();
  const size_t m = (n < max_items) ? n : max_items;
  std::cout << "[";
  for (size_t i = 0; i < m; ++i) {
    if (i) std::cout << ", ";
    std::cout << v[i];
  }
  if (n > m) std::cout << ", ...";
  std::cout << "]\n";
}

inline void print_real_list_inline(const std::string& label, const std::vector<Real>& v, size_t max_items = 10) {
  std::cout << "[fluxpulse_run] " << label << " (n=" << v.size() << "): ";
  const size_t n = v.size();
  const size_t m = (n < max_items) ? n : max_items;
  std::cout << "[";
  for (size_t i = 0; i < m; ++i) {
    if (i) std::cout << ", ";
    std::cout << std::fixed << std::setprecision(6) << static_cast<double>(v[i]);
  }
  if (n > m) std::cout << ", ...";
  std::cout << "]\n";
}

inline void print_pair_couplings(const std::string& label, const std::vector<PairCouplingConfig>& cs, size_t max_items = 10) {
  std::cout << "[fluxpulse_run] " << label << " (n=" << cs.size() << "):\n";
  const size_t n = cs.size();
  const size_t m = (n < max_items) ? n : max_items;
  for (size_t i = 0; i < m; ++i) {
    const auto& c = cs[i];
    std::cout << "  - (" << c.q0 << "," << c.q1 << ") strength=" << std::fixed << std::setprecision(6)
              << static_cast<double>(c.strength) << "\n";
  }
  if (n > m) std::cout << "  ... (" << (n - m) << " more)\n";
}

inline void print_model_summary(const Config& cfg, const ModelBase& model) {
  std::cout << "[fluxpulse_run] ===== Model summary =====\n";
  std::cout << "[fluxpulse_run] model_type: " << cfg.model_type << "\n";
  std::cout << "[fluxpulse_run] hilbert_dim: " << model.hilbert_dim
            << "  subsystems: " << model.subsystem_dims.size() << "\n";
  print_list_inline("subsystem_dims", model.subsystem_dims);

  std::cout << "[fluxpulse_run] terms: drift=" << model.drift_terms.size()
            << "  control=" << model.control_terms.size()
            << "  collapse_ops=" << model.collapse_ops.size() << "\n";

  const std::string type = lower_ascii(cfg.model_type);

  if (type == "generic_qubit" || type == "qubit") {
    std::cout << "[fluxpulse_run] --- generic_qubit parameters ---\n";
    std::cout << "[fluxpulse_run] omega=" << static_cast<double>(cfg.generic_qubit.omega)
              << "  detuning=" << static_cast<double>(cfg.generic_qubit.detuning) << "\n";
    std::cout << "[fluxpulse_run] t1=" << static_cast<double>(cfg.generic_qubit.t1)
              << "  tphi=" << static_cast<double>(cfg.generic_qubit.tphi) << "\n";
  } else if (type == "coupled_qubits" || type == "coupled" || type == "2q" || type == "multi_qubit") {
    std::cout << "[fluxpulse_run] --- coupled_qubits parameters ---\n";
    std::cout << "[fluxpulse_run] nqubits=" << cfg.coupled_qubits.nqubits << "\n";
    if (!cfg.coupled_qubits.omega.empty()) print_real_list_inline("omega", cfg.coupled_qubits.omega);
    if (!cfg.coupled_qubits.detuning.empty()) print_real_list_inline("detuning", cfg.coupled_qubits.detuning);
    if (!cfg.coupled_qubits.t1.empty()) print_real_list_inline("t1", cfg.coupled_qubits.t1);
    if (!cfg.coupled_qubits.tphi.empty()) print_real_list_inline("tphi", cfg.coupled_qubits.tphi);

    std::cout << "[fluxpulse_run] drives: enable_x=" << (cfg.coupled_qubits.enable_drive_x ? "true" : "false")
              << " enable_y=" << (cfg.coupled_qubits.enable_drive_y ? "true" : "false")
              << " enable_z=" << (cfg.coupled_qubits.enable_drive_z ? "true" : "false") << "\n";

    if (!cfg.coupled_qubits.exchange_xy.empty()) print_pair_couplings("exchange_xy", cfg.coupled_qubits.exchange_xy);
    if (!cfg.coupled_qubits.zz.empty()) print_pair_couplings("zz", cfg.coupled_qubits.zz);

    if (!cfg.coupled_qubits.drive_x.empty()) print_list_inline("drive_x ChannelIds", cfg.coupled_qubits.drive_x);
    if (!cfg.coupled_qubits.drive_y.empty()) print_list_inline("drive_y ChannelIds", cfg.coupled_qubits.drive_y);
    if (!cfg.coupled_qubits.drive_z.empty()) print_list_inline("drive_z ChannelIds", cfg.coupled_qubits.drive_z);
  } else if (type == "duffing_transmon" || type == "transmon") {
    std::cout << "[fluxpulse_run] --- duffing_transmon parameters ---\n";
    std::cout << "[fluxpulse_run] levels=" << cfg.duffing_transmon.levels
              << " omega=" << static_cast<double>(cfg.duffing_transmon.omega)
              << " anharm=" << static_cast<double>(cfg.duffing_transmon.anharm)
              << " drive_scale=" << static_cast<double>(cfg.duffing_transmon.drive_scale) << "\n";
    std::cout << "[fluxpulse_run] t1=" << static_cast<double>(cfg.duffing_transmon.t1)
              << " tphi=" << static_cast<double>(cfg.duffing_transmon.tphi) << "\n";
  }

  std::cout << "[fluxpulse_run] =========================\n";
}

inline void print_program_summary(const PulseProgram& program) {
  std::cout << "[fluxpulse_run] ===== Program summary =====\n";
  std::cout << "[fluxpulse_run] pulses=" << program.pulses.size() << "\n";
  if (program.grid.size() > 0) {
    const Real dt = program.grid.is_uniform()
                        ? program.grid.dt
                        : (program.grid.size() > 1 ? program.grid.dt_interval(0) : Real{0});
    std::cout << "[fluxpulse_run] grid: n_points=" << program.grid.size()
              << " dt=" << static_cast<double>(dt)
              << " t0=" << static_cast<double>(program.grid.t(0))
              << " t_end=" << static_cast<double>(program.grid.t_end()) << "\n";
  }
  // Print first few pulses/channels
  const size_t max_p = 12;
  for (size_t i = 0; i < program.pulses.size() && i < max_p; ++i) {
    const auto& p = program.pulses[i];
    std::cout << "  - pulse[" << i << "] channel_id=" << p.channel.value
              << " segments=" << p.envelope.segments.size() << "\n";
  }
  if (program.pulses.size() > max_p) {
    std::cout << "  ... (" << (program.pulses.size() - max_p) << " more pulses)\n";
  }
  std::cout << "[fluxpulse_run] ==========================\n";
}

inline void print_initial_state_summary(const Config& cfg, const ModelBase& model) {
  std::cout << "[fluxpulse_run] ===== Initial state =====\n";
  std::cout << "[fluxpulse_run] type=" << cfg.initial_state.type << "\n";

  std::vector<Dim> dims = cfg.initial_state.dims;
  if (dims.empty()) dims = model.subsystem_dims;

  print_list_inline("dims", dims);

  // Compute linear index for display
  uint64_t total = 1;
  for (Dim d : dims) total *= static_cast<uint64_t>(d);

  uint64_t idx = 0;
  if (cfg.initial_state.has_linear_index) {
    idx = cfg.initial_state.linear_index;
    std::cout << "[fluxpulse_run] linear_index=" << idx << "\n";
  } else {
    // digits little-endian
    const auto& digits = cfg.initial_state.index;
    if (!digits.empty()) print_list_inline("digits(le)", digits);
    uint64_t stride = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      const uint64_t di = digits.empty() ? 0u : static_cast<uint64_t>(digits[i]);
      idx += di * stride;
      stride *= static_cast<uint64_t>(dims[i]);
    }
    std::cout << "[fluxpulse_run] computed_linear_index=" << idx << "\n";
  }

  std::cout << "[fluxpulse_run] hilbert_dim=" << model.hilbert_dim
            << " (dims product=" << total << ")\n";
  std::cout << "[fluxpulse_run] ==========================\n";
}

inline void print_observable_summary(const std::vector<std::string>& names) {
  std::cout << "[fluxpulse_run] ===== Observables =====\n";
  std::cout << "[fluxpulse_run] count=" << names.size() << "\n";
  const size_t max_o = 30;
  for (size_t i = 0; i < names.size() && i < max_o; ++i) {
    std::cout << "  [" << i << "] " << names[i] << "\n";
  }
  if (names.size() > max_o) std::cout << "  ... (" << (names.size() - max_o) << " more)\n";
  std::cout << "[fluxpulse_run] ======================\n";
}

inline void print_result_summary(const Result& r) {
  std::cout << "[fluxpulse_run] ===== Result summary =====\n";
  std::cout << "[fluxpulse_run] observables=" << r.num_observables()
            << "  times=" << r.num_times() << "\n";
  if (r.num_times() > 0) {
    std::cout << "[fluxpulse_run] t0=" << static_cast<double>(r.tlist.front())
              << "  t_end=" << static_cast<double>(r.tlist.back()) << "\n";
  }

  const Dim nobs = r.num_observables();
  const Dim nt = r.num_times();
  const size_t max_o = 20;

  for (Dim o = 0; o < nobs && static_cast<size_t>(o) < max_o; ++o) {
    Real mn = std::numeric_limits<Real>::infinity();
    Real mx = -std::numeric_limits<Real>::infinity();
    for (Dim i = 0; i < nt; ++i) {
      const Real v = r.expect_at(o, i);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const std::string name = (r.observable_names.size() == static_cast<size_t>(nobs))
                                 ? r.observable_names[static_cast<size_t>(o)]
                                 : std::string("obs") + std::to_string(static_cast<int>(o));
    std::cout << "  - " << name
              << "  first=" << std::fixed << std::setprecision(6) << static_cast<double>(r.expect_at(o, 0))
              << "  last=" << std::fixed << std::setprecision(6) << static_cast<double>(r.expect_at(o, nt - 1))
              << "  min=" << std::fixed << std::setprecision(6) << static_cast<double>(mn)
              << "  max=" << std::fixed << std::setprecision(6) << static_cast<double>(mx)
              << "\n";
  }
  if (static_cast<size_t>(nobs) > max_o) {
    std::cout << "  ... (" << (static_cast<size_t>(nobs) - max_o) << " more observables)\n";
  }

  // Optional final pure state vector.
  // This is expected for a single-trajectory run and intentionally absent
  // for multi-trajectory MCWF ensembles.
  if (!r.final_state.empty()) {
    Real norm2 = Real{0};
    for (const Complex& z : r.final_state) {
      norm2 += z.re * z.re + z.im * z.im;
    }

    std::cout << "[fluxpulse_run] final_state: dim="
              << r.final_state.size()
              << "  norm2=" << std::fixed << std::setprecision(10)
              << static_cast<double>(norm2)
              << "\n";
  } else {
    std::cout << "[fluxpulse_run] final_state: not available\n";
  }

  std::cout << "[fluxpulse_run] ==========================\n";
}

Status build_model_from_config(const Config& cfg, ModelBase* out_model) {
  if (!out_model) return Error(ErrorCode::kInvalidArgument, "build_model_from_config: out_model null");

  const std::string type = lower_ascii(cfg.model_type);

  // ---------------- generic_qubit ----------------
  if (type == "generic_qubit" || type == "qubit") {
    effective::GenericQubitParams p;
    p.omega = cfg.generic_qubit.omega;
    p.detuning = cfg.generic_qubit.detuning;
    p.t1 = cfg.generic_qubit.t1;
    p.tphi = cfg.generic_qubit.tphi;

    // Safe default: enable axes only if corresponding pulses exist in program.
    p.enable_drive_x = (cfg.program.find_pulse_index(ChannelId::DriveX()) >= 0);
    p.enable_drive_y = (cfg.program.find_pulse_index(ChannelId::DriveY()) >= 0);
    p.enable_drive_z = (cfg.program.find_pulse_index(ChannelId::DriveZ()) >= 0);

    return effective::build_generic_qubit_model(p, out_model);
  }

  // ---------------- coupled_qubits (Phase 1 multi-qubit) ----------------
  if (type == "coupled_qubits" || type == "coupled" || type == "2q" || type == "multi_qubit") {
    effective::CoupledQubitsParams p;
    p.nqubits = static_cast<Index>(cfg.coupled_qubits.nqubits);

    p.omega = cfg.coupled_qubits.omega;
    p.detuning = cfg.coupled_qubits.detuning;
    p.t1 = cfg.coupled_qubits.t1;
    p.tphi = cfg.coupled_qubits.tphi;

    // Enable controls:
    // - Respect cfg flags, but auto-enable if a pulse exists on any channel of that axis.
    auto has_axis_pulse = [&](uint16_t axis_base) -> bool {
      for (const Pulse& pulse : cfg.program.pulses) {
        const uint16_t id = pulse.channel.value;
        if (id >= axis_base && ((id - axis_base) % 3u) == 0u) return true;
      }
      return false;
    };

    p.enable_drive_x = cfg.coupled_qubits.enable_drive_x || has_axis_pulse(/*axis_base=*/1);
    p.enable_drive_y = cfg.coupled_qubits.enable_drive_y || has_axis_pulse(/*axis_base=*/2);
    p.enable_drive_z = cfg.coupled_qubits.enable_drive_z || has_axis_pulse(/*axis_base=*/3);

    // Optional explicit channels (empty => defaults generated by builder)
    p.drive_x = cfg.coupled_qubits.drive_x;
    p.drive_y = cfg.coupled_qubits.drive_y;
    p.drive_z = cfg.coupled_qubits.drive_z;

    // Couplings
    for (const PairCouplingConfig& c : cfg.coupled_qubits.exchange_xy) {
      effective::QubitPairCoupling cc;
      cc.q0 = static_cast<Index>(c.q0);
      cc.q1 = static_cast<Index>(c.q1);
      cc.strength = c.strength;
      p.exchange_xy.push_back(cc);
    }
    for (const PairCouplingConfig& c : cfg.coupled_qubits.zz) {
      effective::QubitPairCoupling cc;
      cc.q0 = static_cast<Index>(c.q0);
      cc.q1 = static_cast<Index>(c.q1);
      cc.strength = c.strength;
      p.zz.push_back(cc);
    }

    return effective::build_coupled_qubits_model(p, out_model);
  }

  // ---------------- duffing_transmon ----------------
  if (type == "duffing_transmon" || type == "transmon") {
    effective::DuffingTransmonParams p;
    p.levels = cfg.duffing_transmon.levels;
    p.omega = cfg.duffing_transmon.omega;
    p.anharm = cfg.duffing_transmon.anharm;
    p.drive_scale = cfg.duffing_transmon.drive_scale;
    p.t1 = cfg.duffing_transmon.t1;
    p.tphi = cfg.duffing_transmon.tphi;

    // Enable axes only if used by pulses (DriveX/DriveY are qubit-0 channels).
    p.enable_drive_x = (cfg.program.find_pulse_index(ChannelId::DriveX()) >= 0);
    p.enable_drive_y = (cfg.program.find_pulse_index(ChannelId::DriveY()) >= 0);

    return effective::build_duffing_transmon_model(p, out_model);
  }

  return Error(ErrorCode::kInvalidArgument,
               "Unsupported model.type in fluxpulse_run (supported: generic_qubit, coupled_qubits, duffing_transmon)");
}

Status build_default_observable_names(const ModelBase& model, std::vector<std::string>* out_names) {
  if (!out_names) return Error(ErrorCode::kInvalidArgument, "build_default_observable_names: out_names null");
  out_names->clear();

  const std::vector<Dim>& dims = model.subsystem_dims;
  if (dims.size() == 1 && dims[0] == 2) {
    out_names->push_back("Z");
    return Ok();
  }
  // Multi-qubit: default to Z0
  if (!dims.empty() && dims[0] == 2) {
    out_names->push_back("Z0");
    return Ok();
  }
  // Single-mode multi-level: default to number operator
  out_names->push_back("N");
  return Ok();
}

Status build_observables_from_config(const ModelBase& model,
                                    const ObservableConfig& obs_cfg,
                                    std::vector<DenseOp>* out_ops,
                                    std::vector<std::string>* out_names) {
  if (!out_ops || !out_names) return Error(ErrorCode::kInvalidArgument, "build_observables_from_config: null out");

  std::vector<std::string> names = obs_cfg.names;
  if (names.empty()) {
    FLUXPULSE_RETURN_IF_ERROR(build_default_observable_names(model, &names));
  }

  std::vector<DenseOp> ops;
  FLUXPULSE_RETURN_IF_ERROR(sim::build_observables(names, model.subsystem_dims, &ops));

  *out_ops = std::move(ops);
  *out_names = std::move(names);
  return Ok();
}

Status build_initial_state_vector(const Config& cfg, const ModelBase& model, std::vector<Complex>* out_psi0) {
  if (!out_psi0) return Error(ErrorCode::kInvalidArgument, "build_initial_state_vector: out_psi0 null");
  out_psi0->assign(static_cast<size_t>(model.hilbert_dim), complex_zero());

  // Phase 1: basis initialization
  if (lower_ascii(cfg.initial_state.type) != "basis") {
    return Error(ErrorCode::kInvalidArgument, "initial_state.type: Phase 1 supports only 'basis'");
  }

  // Prefer dims from config; otherwise use model dims.
  std::vector<Dim> dims = cfg.initial_state.dims;
  if (dims.empty()) dims = model.subsystem_dims;

  // Basic validation: product(dims) matches hilbert_dim.
  uint64_t total = 1;
  for (Dim d : dims) {
    if (d <= 0) return Error(ErrorCode::kInvalidArgument, "initial_state.dims contains non-positive dimension");
    total *= static_cast<uint64_t>(d);
  }
  if (total != static_cast<uint64_t>(model.hilbert_dim)) {
    return Error(ErrorCode::kInvalidArgument, "initial_state.dims product does not match model.hilbert_dim");
  }

  uint64_t idx = 0;

  if (cfg.initial_state.has_linear_index) {
    idx = cfg.initial_state.linear_index;
    if (idx >= total) return Error(ErrorCode::kOutOfRange, "initial_state.linear_index out of range");
  } else {
    // Digits are little-endian: index digits [subsystem0, subsystem1, ...]
    const std::vector<Dim>& digits = cfg.initial_state.index;
    if (!digits.empty() && digits.size() != dims.size()) {
      return Error(ErrorCode::kInvalidArgument, "initial_state.index length must match dims length");
    }
    uint64_t stride = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      const uint64_t di = digits.empty() ? 0u : static_cast<uint64_t>(digits[i]);
      if (di >= static_cast<uint64_t>(dims[i])) {
        return Error(ErrorCode::kOutOfRange, "initial_state.index digit out of range");
      }
      idx += di * stride;
      stride *= static_cast<uint64_t>(dims[i]);
    }
    if (idx >= total) return Error(ErrorCode::kOutOfRange, "initial_state.index out of range");
  }

  (*out_psi0)[static_cast<size_t>(idx)] = complex_one();
  return Ok();
}

Status write_outputs(const Config& cfg,
                     const Result& r,
                     const std::vector<io::TraceEvent>* trace_events) {
  // Result
  std::string result_path = cfg.output.result_path;
  if (result_path.empty()) result_path = "out/result.json";

  if (ends_with(result_path, ".csv")) {
    FLUXPULSE_RETURN_IF_ERROR(io::write_result_csv(result_path, r));
  } else {
    FLUXPULSE_RETURN_IF_ERROR(io::write_result_json(result_path, r));
  }

  // Trace
  if (cfg.output.write_trace) {
    std::string trace_path = cfg.output.trace_path;
    if (trace_path.empty()) trace_path = "out/trace.json";
    if (!trace_events) return Error(ErrorCode::kInvalidArgument, "write_outputs: trace_events null");
    FLUXPULSE_RETURN_IF_ERROR(io::write_chrome_trace_json(trace_path, *trace_events));
  }

  return Ok();
}

}  // namespace
}  // namespace fluxpulse

int main(int argc, char** argv) {
  using namespace fluxpulse;

  if (argc < 2) {
    std::cerr << "Usage: fluxpulse_run <config.json> [--compiled_program <compiled.json>]\n";
    return 2;
  }

  std::string config_path;
  std::string compiled_program_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: fluxpulse_run <config.json> [--compiled_program <compiled.json>]\n";
      return 0;
    }

    if (arg == "--compiled_program") {
      if (i + 1 >= argc) {
        std::cerr << "ERROR: --compiled_program requires a path argument\n";
        return 2;
      }
      compiled_program_path = std::string(argv[++i]);
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      std::cerr << "ERROR: unknown option: " << arg << "\n";
      std::cerr << "Usage: fluxpulse_run <config.json> [--compiled_program <compiled.json>]\n";
      return 2;
    }

    if (config_path.empty()) {
      config_path = arg;
      continue;
    }

    std::cerr << "ERROR: unexpected extra positional argument: " << arg << "\n";
    return 2;
  }

  if (config_path.empty()) {
    std::cerr << "ERROR: missing <config.json>\n";
    std::cerr << "Usage: fluxpulse_run <config.json> [--compiled_program <compiled.json>]\n";
    return 2;
  }

  std::cout << "[fluxpulse_run] config: " << config_path << "\n";
  if (!compiled_program_path.empty()) {
    std::cout << "[fluxpulse_run] compiled_program: " << compiled_program_path << "\n";
  }


  // Load config
  Config cfg;
  size_t error_pos = 0;
  Status st;
  {
    ScopedStage stage("Loading config");
    st = load_config_file(config_path, &cfg, &error_pos);
  }
  if (!st.ok()) {
    std::cerr << "ERROR: config load failed.\n";
    std::cerr << "  path: " << config_path << "\n";
    std::cerr << "  error_pos: " << static_cast<unsigned long long>(error_pos) << "\n";
    return 1;
  }

  // Optional: override cfg.program using a Phase-2 compiled program JSON.
  if (!compiled_program_path.empty()) {
    PulseProgram compiled_prog;
    size_t prog_error_pos = 0;
    Status st_prog;
    {
      ScopedStage stage("Loading compiled program");
      st_prog = load_compiled_program_json_file(compiled_program_path, cfg, &compiled_prog, &prog_error_pos);
    }
    if (!st_prog.ok()) {
      std::cerr << "ERROR: compiled program load/convert failed.\n";
      std::cerr << "  path: " << compiled_program_path << "\n";
      // error_pos is meaningful for JSON parse failures; for semantic conversion errors it may remain 0.
      std::cerr << "  error_pos: " << prog_error_pos << "\n";
      std::cerr << "  message: " << st_prog.message << "\n";
      return 1;
    }
    cfg.program = std::move(compiled_prog);
  }

  // Build model from cfg
  ModelBase model;
  {
    ScopedStage stage("Building model");
    st = fluxpulse::build_model_from_config(cfg, &model);
  }
  if (!st.ok()) {
    std::cerr << "ERROR: model build failed.\n";
    return 1;
  }

  // Detailed summaries
  print_model_summary(cfg, model);
  print_program_summary(cfg.program);

  // Build observables
  std::vector<DenseOp> obs_ops;
  std::vector<std::string> obs_names;
  {
    ScopedStage stage("Building observables");
    st = fluxpulse::build_observables_from_config(model, cfg.observables, &obs_ops, &obs_names);
  }
  if (!st.ok()) {
    std::cerr << "ERROR: observable build failed.\n";
    return 1;
  }

  print_observable_summary(obs_names);

  // Initial state vector
  std::vector<Complex> psi0;
  {
    ScopedStage stage("Building initial state");
    st = fluxpulse::build_initial_state_vector(cfg, model, &psi0);
  }
  if (!st.ok()) {
    std::cerr << "ERROR: initial_state build failed.\n";
    return 1;
  }

  print_initial_state_summary(cfg, model);

  // Processor
  Processor proc;
  proc.model = std::move(model);
  proc.program = cfg.program;
  proc.observables = std::move(obs_ops);
  proc.observable_names = std::move(obs_names);

  // Run options
  proc.opt.trajectories = cfg.run.trajectories;
  proc.opt.base_seed = static_cast<Seed>(cfg.run.base_seed);
  proc.opt.sample_stride = cfg.run.sample_stride;
  proc.opt.renormalize_each_step = cfg.run.renormalize_each_step;

  // Summary
  std::cout << "[fluxpulse_run] model_type: " << cfg.model_type
            << "  hilbert_dim: " << proc.model.hilbert_dim
            << "  subsystems: " << proc.model.subsystem_dims.size() << "\n";
  if (proc.program.grid.size() > 0) {
    std::cout << "[fluxpulse_run] grid: n_points=" << proc.program.grid.size()
              << "  dt=" << (proc.program.grid.is_uniform() ? proc.program.grid.dt
                              : (proc.program.grid.size() > 1 ? proc.program.grid.dt_interval(0) : Real{0}))
              << "  t_final=" << proc.program.grid.t_end() << "\n";
  }
  std::cout << "[fluxpulse_run] trajectories=" << proc.opt.trajectories
            << "  sample_stride=" << proc.opt.sample_stride
            << "  pulses=" << proc.program.pulses.size()
            << "  observables=" << proc.observables.size() << "\n";
  std::cout << "[fluxpulse_run] trace: write_trace=" << (cfg.output.write_trace ? "true" : "false")
            << "  emit_segments=" << (cfg.trace.emit_segments ? "true" : "false")
            << "  emit_counters=" << (cfg.trace.emit_counters ? "true" : "false")
            << "  counter_stride=" << cfg.trace.counter_stride
            << "\n";

  // Optional MCWF diagnostics (for trace counters)
  mcwf::McwfDiagnostics diag;
  if (cfg.output.write_trace && cfg.trace.emit_counters && proc.opt.trajectories == 1) {
    proc.opt.diagnostics = &diag;
    proc.opt.diagnostic_stride = cfg.trace.diagnostic_stride;
  } else {
    proc.opt.diagnostics = nullptr;
    proc.opt.diagnostic_stride = 1;
  }

  // Run
  Result result;
  {
    ScopedStage stage("Running simulation");

    const uint64_t n_tr = proc.opt.trajectories;
    if (n_tr <= 1) {
      st = proc.run_mcwf(Span<const Complex>(psi0.data(), static_cast<Dim>(psi0.size())), &result);
      if (!st.ok()) {
        std::cerr << "ERROR: simulation failed.\n";
        return 1;
      }
    } else {
      // Custom accumulate loop with per-trajectory progress.
      st = proc.validate();
      if (!st.ok()) {
        std::cerr << "ERROR: processor validation failed.\n";
        return 1;
      }

      const Dim n_obs = static_cast<Dim>(proc.observables.size());
      const Dim n_points = proc.program.grid.size();
      const Dim stride = (proc.opt.sample_stride <= 0) ? 1 : proc.opt.sample_stride;
      const Dim n_samples = compute_num_samples(n_points, stride);

      mcwf::McwfWorkspace ws;
      st = ws.init(proc.model, proc.program, n_obs);
      if (!st.ok()) {
        std::cerr << "ERROR: mcwf workspace init failed.\n";
        return 1;
      }

      mcwf::McwfOptions mopt;
      mopt.sample_stride = proc.opt.sample_stride;
      mopt.renormalize_each_step = proc.opt.renormalize_each_step;

      // Diagnostics are disabled for multi-trajectory runs (trace counters are most meaningful for 1 traj).
      mopt.diagnostic_stride = 1;
      mopt.diagnostics = nullptr;

      // A multi-trajectory MCWF run represents an ensemble and has no unique
      // final pure state vector. Do not capture one trajectory as "the" state.
      mopt.final_state = nullptr;

      std::vector<Real> sum(static_cast<size_t>(n_obs) * static_cast<size_t>(n_samples), Real{0});
      std::vector<Real> sum_sq(static_cast<size_t>(n_obs) * static_cast<size_t>(n_samples), Real{0});

      // Reuse a per-trajectory output buffer to avoid reallocations.
      Result traj_out;
      traj_out.resize(n_obs, n_samples);

      const auto start_time = std::chrono::steady_clock::now();
      const uint64_t step = (n_tr >= 50) ? (n_tr / 50) : 1;  // <= ~50 updates

      for (uint64_t tr = 0; tr < n_tr; ++tr) {
        const Seed seed = derive_trajectory_seed(proc.opt.base_seed, static_cast<TrajectoryId>(tr));
        Rng rng;
        rng.init_splitmix64();
        rng.seed(seed);

        st = mcwf::run_mcwf_trajectory(
            proc.model, proc.program,
            Span<const DenseOp>(proc.observables.data(), n_obs),
            Span<const Complex>(psi0.data(), static_cast<Dim>(psi0.size())),
            &rng, mopt, &ws,
            &traj_out);
        if (!st.ok()) {
          std::cerr << "ERROR: simulation failed in trajectory " << tr << ".\n";
          return 1;
        }

        // Accumulate first and second moments (for mean/variance/stderr).
        for (size_t i = 0; i < sum.size(); ++i) {
          const Real x = traj_out.expectations[i];
          sum[i] += x;
          sum_sq[i] += x * x;
        }

        if (((tr + 1) % step) == 0 || (tr + 1) == n_tr) {
          print_progress(tr + 1, n_tr, start_time, (tr + 1) == n_tr);
        }
      }

      // Finalize Result (mean + uncertainty)
      result.resize_with_uncertainty(n_obs, n_samples);

      // Fill tlist consistent with solver sampling (include last point even if not on stride).
      Dim pos = 0;
      for (Dim i = 0; i < n_points; i += stride) {
        result.tlist[static_cast<size_t>(pos++)] = proc.program.grid.t(i);
      }
      const Dim last = n_points - 1;
      if ((last % stride) != 0) {
        result.tlist[static_cast<size_t>(pos++)] = proc.program.grid.t(last);
      }

      const Real invN = Real{1} / static_cast<Real>(n_tr);
      for (size_t i = 0; i < sum.size(); ++i) {
        const Real mean = sum[i] * invN;
        Real var = sum_sq[i] * invN - mean * mean;  // population variance
        if (var < Real{0}) var = Real{0};
        result.expectations[i] = mean;
        result.expectations_var[i] = var;
        result.expectations_stderr[i] = std::sqrt(var * invN);
      }
      result.observable_names = proc.observable_names;
    }
  }


  // Attach observable names for writers
  result.observable_names = proc.observable_names;

  // Trace events (optional)
  std::vector<io::TraceEvent> events;
  if (cfg.output.write_trace) {
    ScopedStage stage("Building trace");
    const Real scale_to_us = cfg.trace.time_scale_to_us;

    // Always include a solver span so traces are non-empty even with no pulses (e.g., T1)
    st = io::append_run_span_event(proc.program, scale_to_us, "mcwf_run", &events);
    if (!st.ok()) {
      std::cerr << "ERROR: trace run span failed.\n";
      return 1;
    }

    if (cfg.trace.emit_segments) {
      st = io::append_pulse_segment_events(proc.program, scale_to_us, "pulse:", &events);
      if (!st.ok()) {
        std::cerr << "ERROR: trace segment export failed.\n";
        return 1;
      }
    }

    if (cfg.trace.emit_counters) {
      st = io::append_pulse_counter_events(proc.program, cfg.trace.pulse_counter_stride, scale_to_us, "pulse:", &events);
      if (!st.ok()) {
        std::cerr << "ERROR: trace counter export failed.\n";
        return 1;
      }
    }

    // MCWF diagnostics counters (norm2, p_total, jumped, jump_index)
    if (proc.opt.diagnostics && !diag.tlist.empty()) {
      st = io::append_mcwf_diagnostic_counter_events(
          Span<const Real>(diag.tlist.data(), static_cast<Dim>(diag.tlist.size())),
          Span<const Real>(diag.norm2.data(), static_cast<Dim>(diag.norm2.size())),
          Span<const Real>(diag.p_total.data(), static_cast<Dim>(diag.p_total.size())),
          Span<const int32_t>(diag.jumped.data(), static_cast<Dim>(diag.jumped.size())),
          Span<const int32_t>(diag.jump_index.data(), static_cast<Dim>(diag.jump_index.size())),
          scale_to_us,
          "mcwf",
          /*tid=*/9999,
          &events);
      if (!st.ok()) {
        std::cerr << "ERROR: trace mcwf diagnostics export failed.\n";
        return 1;
      }
    }
  }

  // Write outputs
  {
    ScopedStage stage("Writing outputs");
    st = fluxpulse::write_outputs(cfg, result, &events);
  }
  if (!st.ok()) {
    std::cerr << "ERROR: output write failed.\n";
    return 1;
  }

  print_result_summary(result);
  std::cout << "[fluxpulse_run] wrote result: " << (cfg.output.result_path.empty() ? std::string("out/result.json") : cfg.output.result_path) << "\n";
  if (cfg.output.write_trace) {
    std::cout << "[fluxpulse_run] wrote trace: " << (cfg.output.trace_path.empty() ? std::string("out/trace.json") : cfg.output.trace_path) << "\n";
  }

  std::cout << "[fluxpulse_run] OK\n";
  return 0;
}
