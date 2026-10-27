// apps/fluxpulse_run_mpi.cpp
#include <cstdint>
#include <cctype>
#include <charconv>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/device/processor.hpp"
#include "fluxpulse/io/config_loader.hpp"
#include "fluxpulse/io/pulse_program_reader.hpp"
#include "fluxpulse/io/result_writer.hpp"
#include "fluxpulse/model/effective/coupled_qubits.hpp"
#include "fluxpulse/model/effective/duffing_transmon.hpp"
#include "fluxpulse/model/effective/generic_qubit.hpp"
#include "fluxpulse/mpi/mpi_context.hpp"
#include "fluxpulse/mpi/mpi_runner.hpp"
#include "fluxpulse/mpi/mpi_types.hpp"
#include "fluxpulse/sim/observable_builders.hpp"
#include "fluxpulse/sim/result.hpp"

namespace fluxpulse {
namespace {

inline bool ends_with(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) return false;
  return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lower_ascii(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

inline Dim compute_num_samples(Dim n_points, Dim stride) {
  if (n_points <= 0) return 0;
  if (stride <= 0) stride = 1;
  const Dim last = n_points - 1;
  Dim n = last / stride + 1;
  if ((last % stride) != 0) n += 1;
  return n;
}

void print_usage() {
  std::cout
      << "Usage: fluxpulse_run_mpi <config.json> "
      << "[--compiled_program <compiled.json>] "
      << "[--batch_size <N>] [--static | --dynamic]\n";
}

Status build_model_from_config(const Config& cfg, ModelBase* out_model) {
  if (!out_model) {
    return Error(ErrorCode::kInvalidArgument,
                 "build_model_from_config: out_model null");
  }

  const std::string type = lower_ascii(cfg.model_type);

  if (type == "generic_qubit" || type == "qubit") {
    effective::GenericQubitParams p;
    p.omega = cfg.generic_qubit.omega;
    p.detuning = cfg.generic_qubit.detuning;
    p.t1 = cfg.generic_qubit.t1;
    p.tphi = cfg.generic_qubit.tphi;

    p.enable_drive_x =
        (cfg.program.find_pulse_index(ChannelId::DriveX()) >= 0);
    p.enable_drive_y =
        (cfg.program.find_pulse_index(ChannelId::DriveY()) >= 0);
    p.enable_drive_z =
        (cfg.program.find_pulse_index(ChannelId::DriveZ()) >= 0);

    return effective::build_generic_qubit_model(p, out_model);
  }

  if (type == "coupled_qubits" || type == "coupled" ||
      type == "2q" || type == "multi_qubit") {
    effective::CoupledQubitsParams p;
    p.nqubits = static_cast<Index>(cfg.coupled_qubits.nqubits);

    p.omega = cfg.coupled_qubits.omega;
    p.detuning = cfg.coupled_qubits.detuning;
    p.t1 = cfg.coupled_qubits.t1;
    p.tphi = cfg.coupled_qubits.tphi;

    auto has_axis_pulse = [&](uint16_t axis_base) -> bool {
      for (const Pulse& pulse : cfg.program.pulses) {
        const uint16_t id = pulse.channel.value;
        if (id >= axis_base && ((id - axis_base) % 3u) == 0u) return true;
      }
      return false;
    };

    p.enable_drive_x =
        cfg.coupled_qubits.enable_drive_x || has_axis_pulse(1);
    p.enable_drive_y =
        cfg.coupled_qubits.enable_drive_y || has_axis_pulse(2);
    p.enable_drive_z =
        cfg.coupled_qubits.enable_drive_z || has_axis_pulse(3);

    p.drive_x = cfg.coupled_qubits.drive_x;
    p.drive_y = cfg.coupled_qubits.drive_y;
    p.drive_z = cfg.coupled_qubits.drive_z;

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

  if (type == "duffing_transmon" || type == "transmon") {
    effective::DuffingTransmonParams p;
    p.levels = cfg.duffing_transmon.levels;
    p.omega = cfg.duffing_transmon.omega;
    p.anharm = cfg.duffing_transmon.anharm;
    p.drive_scale = cfg.duffing_transmon.drive_scale;
    p.t1 = cfg.duffing_transmon.t1;
    p.tphi = cfg.duffing_transmon.tphi;

    p.enable_drive_x =
        (cfg.program.find_pulse_index(ChannelId::DriveX()) >= 0);
    p.enable_drive_y =
        (cfg.program.find_pulse_index(ChannelId::DriveY()) >= 0);

    return effective::build_duffing_transmon_model(p, out_model);
  }

  return Error(
      ErrorCode::kInvalidArgument,
      "Unsupported model.type in fluxpulse_run_mpi "
      "(supported: generic_qubit, coupled_qubits, duffing_transmon)");
}

Status build_default_observable_names(const ModelBase& model,
                                      std::vector<std::string>* out_names) {
  if (!out_names) {
    return Error(ErrorCode::kInvalidArgument,
                 "build_default_observable_names: out_names null");
  }
  out_names->clear();

  const std::vector<Dim>& dims = model.subsystem_dims;
  if (dims.size() == 1 && dims[0] == 2) {
    out_names->push_back("Z");
    return Ok();
  }
  if (!dims.empty() && dims[0] == 2) {
    out_names->push_back("Z0");
    return Ok();
  }
  out_names->push_back("N");
  return Ok();
}

Status build_observables_from_config(const ModelBase& model,
                                     const ObservableConfig& obs_cfg,
                                     std::vector<DenseOp>* out_ops,
                                     std::vector<std::string>* out_names) {
  if (!out_ops || !out_names) {
    return Error(ErrorCode::kInvalidArgument,
                 "build_observables_from_config: null out");
  }

  std::vector<std::string> names = obs_cfg.names;
  if (names.empty()) {
    FLUXPULSE_RETURN_IF_ERROR(
        build_default_observable_names(model, &names));
  }

  std::vector<DenseOp> ops;
  FLUXPULSE_RETURN_IF_ERROR(
      sim::build_observables(names, model.subsystem_dims, &ops));

  *out_ops = std::move(ops);
  *out_names = std::move(names);
  return Ok();
}

Status build_initial_state_vector(const Config& cfg,
                                  const ModelBase& model,
                                  std::vector<Complex>* out_psi0) {
  if (!out_psi0) {
    return Error(ErrorCode::kInvalidArgument,
                 "build_initial_state_vector: out_psi0 null");
  }
  out_psi0->assign(static_cast<size_t>(model.hilbert_dim), complex_zero());

  if (lower_ascii(cfg.initial_state.type) != "basis") {
    return Error(ErrorCode::kInvalidArgument,
                 "initial_state.type: Phase 1 supports only 'basis'");
  }

  std::vector<Dim> dims = cfg.initial_state.dims;
  if (dims.empty()) dims = model.subsystem_dims;

  uint64_t total = 1;
  for (Dim d : dims) {
    if (d <= 0) {
      return Error(ErrorCode::kInvalidArgument,
                   "initial_state.dims contains non-positive dimension");
    }
    total *= static_cast<uint64_t>(d);
  }
  if (total != static_cast<uint64_t>(model.hilbert_dim)) {
    return Error(ErrorCode::kInvalidArgument,
                 "initial_state.dims product does not match model.hilbert_dim");
  }

  uint64_t idx = 0;
  if (cfg.initial_state.has_linear_index) {
    idx = cfg.initial_state.linear_index;
    if (idx >= total) {
      return Error(ErrorCode::kOutOfRange,
                   "initial_state.linear_index out of range");
    }
  } else {
    const std::vector<Dim>& digits = cfg.initial_state.index;
    if (!digits.empty() && digits.size() != dims.size()) {
      return Error(ErrorCode::kInvalidArgument,
                   "initial_state.index length must match dims length");
    }

    uint64_t stride = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      const uint64_t di =
          digits.empty() ? 0u : static_cast<uint64_t>(digits[i]);
      if (di >= static_cast<uint64_t>(dims[i])) {
        return Error(ErrorCode::kOutOfRange,
                     "initial_state.index digit out of range");
      }
      idx += di * stride;
      stride *= static_cast<uint64_t>(dims[i]);
    }
    if (idx >= total) {
      return Error(ErrorCode::kOutOfRange,
                   "initial_state.index out of range");
    }
  }

  (*out_psi0)[static_cast<size_t>(idx)] = complex_one();
  return Ok();
}

Status write_result_only(const Config& cfg, const Result& r) {
  std::string result_path = cfg.output.result_path;
  if (result_path.empty()) result_path = "out/result.json";

  if (ends_with(result_path, ".csv")) {
    return io::write_result_csv(result_path, r);
  }
  return io::write_result_json(result_path, r);
}

void print_result_summary(const Result& r) {
  std::cout << "[fluxpulse_run_mpi] observables=" << r.num_observables()
            << "  times=" << r.num_times() << "\n";
  if (r.num_times() > 0) {
    std::cout << "[fluxpulse_run_mpi] t0=" << static_cast<double>(r.tlist.front())
              << "  t_end=" << static_cast<double>(r.tlist.back()) << "\n";
  }
}

}  // namespace
}  // namespace fluxpulse

int main(int argc, char** argv) {
  using namespace fluxpulse;

  mpi::MpiContext mpi_ctx(argc, argv);
  if (!mpi_ctx.valid()) {
    std::cerr << "ERROR: MPI initialization failed.\n";
    return 1;
  }

  std::string config_path;
  std::string compiled_program_path;
  Index batch_size = 1;
  bool force_static = false;
  bool force_dynamic = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      if (mpi_ctx.is_root()) print_usage();
      return 0;
    }

    if (arg == "--compiled_program") {
      if (i + 1 >= argc) {
        if (mpi_ctx.is_root()) {
          std::cerr << "ERROR: --compiled_program requires a path argument\n";
        }
        return 2;
      }
      compiled_program_path = std::string(argv[++i]);
      continue;
    }

    if (arg == "--batch_size") {
      if (i + 1 >= argc) {
        if (mpi_ctx.is_root()) {
          std::cerr << "ERROR: --batch_size requires an integer argument\n";
        }
        return 2;
      }
      const std::string value(argv[++i]);
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), batch_size);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || batch_size <= 0) {
        if (mpi_ctx.is_root()) {
          std::cerr << "ERROR: --batch_size requires a positive integer\n";
        }
        return 2;
      }
      continue;
    }

    if (arg == "--static") {
      force_static = true;
      continue;
    }
    if (arg == "--dynamic") {
      force_dynamic = true;
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      if (mpi_ctx.is_root()) {
        std::cerr << "ERROR: unknown option: " << arg << "\n";
        print_usage();
      }
      return 2;
    }

    if (config_path.empty()) {
      config_path = arg;
      continue;
    }

    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: unexpected extra positional argument: " << arg << "\n";
    }
    return 2;
  }

  if (config_path.empty()) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: missing <config.json>\n";
      print_usage();
    }
    return 2;
  }

  if (force_static && force_dynamic) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: choose only one of --static or --dynamic\n";
    }
    return 2;
  }

  if (mpi_ctx.is_root()) {
    std::cout << "[fluxpulse_run_mpi] config: " << config_path << "\n";
    if (!compiled_program_path.empty()) {
      std::cout << "[fluxpulse_run_mpi] compiled_program: "
                << compiled_program_path << "\n";
    }
    std::cout << "[fluxpulse_run_mpi] world_size: " << mpi_ctx.size() << "\n";
  }

  Config cfg;
  size_t error_pos = 0;
  Status st = load_config_file(config_path, &cfg, &error_pos);
  if (!st.ok()) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: config load failed.\n";
      std::cerr << "  path: " << config_path << "\n";
      std::cerr << "  error_pos: "
                << static_cast<unsigned long long>(error_pos) << "\n";
    }
    return 1;
  }

  if (!compiled_program_path.empty()) {
    PulseProgram compiled_prog;
    size_t prog_error_pos = 0;
    Status st_prog = load_compiled_program_json_file(
        compiled_program_path, cfg, &compiled_prog, &prog_error_pos);
    if (!st_prog.ok()) {
      if (mpi_ctx.is_root()) {
        std::cerr << "ERROR: compiled program load/convert failed.\n";
        std::cerr << "  path: " << compiled_program_path << "\n";
        std::cerr << "  error_pos: " << prog_error_pos << "\n";
        std::cerr << "  message: " << st_prog.message << "\n";
      }
      return 1;
    }
    cfg.program = std::move(compiled_prog);
  }

  ModelBase model;
  st = build_model_from_config(cfg, &model);
  if (!st.ok()) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: model build failed.\n";
    }
    return 1;
  }

  std::vector<DenseOp> obs_ops;
  std::vector<std::string> obs_names;
  st = build_observables_from_config(model, cfg.observables, &obs_ops, &obs_names);
  if (!st.ok()) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: observable build failed.\n";
    }
    return 1;
  }

  std::vector<Complex> psi0;
  st = build_initial_state_vector(cfg, model, &psi0);
  if (!st.ok()) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: initial_state build failed.\n";
    }
    return 1;
  }

  Processor proc;
  proc.model = std::move(model);
  proc.program = cfg.program;
  proc.observables = std::move(obs_ops);
  proc.observable_names = std::move(obs_names);

  proc.opt.trajectories = cfg.run.trajectories;
  proc.opt.base_seed = static_cast<Seed>(cfg.run.base_seed);
  proc.opt.sample_stride = cfg.run.sample_stride;
  proc.opt.renormalize_each_step = cfg.run.renormalize_each_step;

  // Multi-rank ensemble path: diagnostics/trace disabled for now.
  proc.opt.diagnostics = nullptr;
  proc.opt.diagnostic_stride = 1;

  if (mpi_ctx.is_root()) {
    const Dim n_points = proc.program.grid.size();
    const Dim stride =
        (proc.opt.sample_stride <= 0) ? 1 : proc.opt.sample_stride;
    const Dim n_samples = compute_num_samples(n_points, stride);

    std::cout << "[fluxpulse_run_mpi] model_type: " << cfg.model_type
              << "  hilbert_dim: " << proc.model.hilbert_dim
              << "  subsystems: " << proc.model.subsystem_dims.size() << "\n";

    if (n_points > 0) {
      std::cout << "[fluxpulse_run_mpi] grid: n_points=" << n_points
                << "  t_final=" << proc.program.grid.t_end()
                << "  sampled_points=" << n_samples << "\n";
    }

    std::cout << "[fluxpulse_run_mpi] trajectories=" << proc.opt.trajectories
              << "  sample_stride=" << proc.opt.sample_stride
              << "  batch_size=" << batch_size
              << "  observables=" << proc.observables.size()
              << "\n";

    if (cfg.output.write_trace) {
      std::cout << "[fluxpulse_run_mpi] note: trace output is not enabled yet "
                   "in the MPI app; only result output will be written.\n";
    }
  }

  mpi::MpiRunInputs in;
  in.processor = &proc;
  in.psi0 = Span<const Complex>(psi0.data(), static_cast<Dim>(psi0.size()));
  in.mpi.total_trajectories =
      static_cast<TrajectoryId>(proc.opt.trajectories);
  in.mpi.batch_size = batch_size;
  in.mpi.base_seed = proc.opt.base_seed;
  in.mpi.enable_trace = false;
  in.mpi.trace_rank = -1;

  if (force_static) {
    in.mpi.reproducible = true;
    in.mpi.scheduling = mpi::SchedulingMode::kStatic;
  } else {
    in.mpi.reproducible = false;
    in.mpi.scheduling = mpi::SchedulingMode::kDynamic;
  }
  if (force_dynamic) {
    in.mpi.reproducible = false;
    in.mpi.scheduling = mpi::SchedulingMode::kDynamic;
  }

  mpi::MpiRunner runner(&mpi_ctx);
  mpi::MpiRunSummary summary;
  Result result;

  st = runner.Run(in, mpi_ctx.is_root() ? &result : nullptr, &summary);
  if (!st.ok()) {
    if (mpi_ctx.is_root()) {
      std::cerr << "ERROR: MPI simulation failed.\n";
      std::cerr << "  message: " << st.message << "\n";
    }
    return 1;
  }

  if (mpi_ctx.is_root()) {
    result.observable_names = proc.observable_names;

    st = write_result_only(cfg, result);
    if (!st.ok()) {
      std::cerr << "ERROR: output write failed.\n";
      return 1;
    }

    print_result_summary(result);

    std::cout << "[fluxpulse_run_mpi] wrote result: "
              << (cfg.output.result_path.empty()
                      ? std::string("out/result.json")
                      : cfg.output.result_path)
              << "\n";
    std::cout << "[fluxpulse_run_mpi] OK\n";
  }

  return 0;
}
