// include/fluxpulse/io/config_types.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/types.hpp"

namespace fluxpulse {

// This header defines "plain data" config structs loaded from JSON.
//
// Phase 1 supported models:
// - generic_qubit
// - coupled_qubits (multi-qubit 2-level with exchange/ZZ couplings)
//
// Additional model blocks are included to keep schema stable for later phases.

struct RunOptionsConfig {
  // Optional: if not provided, inferred from program.grid (dt and t_final).
  Real dt = Real{0};
  Real t_final = Real{0};

  // MCWF / sampling controls.
  std::uint64_t trajectories = 0;
  std::uint64_t base_seed = 0;

  // Sample every k-th grid point when writing results/observables (>=1).
  Dim sample_stride = 1;

  // MCWF solver behavior.
  bool renormalize_each_step = true;
};

struct OutputConfig {
  std::string result_path;  // e.g., "out/result.json"
  std::string trace_path;   // e.g., "out/trace.json"
  bool write_trace = false;
};

struct ObservableConfig {
  // Strings like "X", "Y", "Z", "N", "Z0", "Z1", "ZZ01", "P:10", etc.
  std::vector<std::string> names;
};

struct TraceConfig {
  // If true, emit slice events per segment.
  bool emit_segments = true;

  // If true, emit amplitude samples as counters (optional; can be expensive).
  bool emit_counters = false;

  // Backward-compat alias (legacy configs):
  // If set in JSON, used as the default for both pulse_counter_stride and diagnostic_stride
  // unless those are explicitly provided.
  Dim counter_stride = 1;

  // If emit_counters, write one pulse amplitude sample every `pulse_counter_stride` grid points.
  Dim pulse_counter_stride = 1;

  // If MCWF diagnostics are enabled (typically only for 1 trajectory with tracing),
  // record diagnostics every `diagnostic_stride` grid points.
  Dim diagnostic_stride = 1;

  // Scale factor applied when writing time to the trace as "us".
  // Used as: t_us = t_sim * time_scale_to_us, then converted to ns in the trace writer.
  Real time_scale_to_us = Real{1};
};

// -------------------- Initial state config --------------------
//
// Phase 1 supports basis-state initialization.
//
// If omitted, defaults to the all-zero computational basis state,
// with dims inferred from model.
struct InitialStateConfig {
  // Supported (Phase 1): "basis"
  std::string type = "basis";

  // Optional subsystem dims. If empty, loader infers from model.
  std::vector<Dim> dims;

  // For type=="basis":
  // - Provide `index` as per-subsystem digits in little-endian order:
  //     |q_{N-1} ... q1 q0> -> index digits [q0, q1, ..., q_{N-1}]
  // - Or provide `linear_index` (flattened basis index).
  std::vector<Dim> index;

  std::uint64_t linear_index = 0;
  bool has_linear_index = false;
};

// -------------------- Model configs --------------------

// Phase 1: Generic single-qubit (2-level) effective model.
struct GenericQubitModelConfig {
  // Drift: H0 = 0.5 * (omega + detuning) * Z
  Real omega = Real{0};
  Real detuning = Real{0};

  // Optional noise parameters (<=0 disables):
  // Convention: if enabled, rate gamma = 1/T.
  Real t1 = Real{0};    // amplitude damping time constant
  Real tphi = Real{0};  // pure dephasing time constant
};

// Pair coupling entry for multi-qubit 2-level models.
struct PairCouplingConfig {
  Dim q0 = 0;
  Dim q1 = 1;
  Real strength = Real{0};
};

// Phase 1: Multi-qubit (2-level per qubit) effective model with couplings.
struct CoupledQubitsModelConfig {
  Dim nqubits = 2;

  // Per-qubit drift params (either empty or size==nqubits).
  std::vector<Real> omega;
  std::vector<Real> detuning;

  // Optional per-qubit noise times (either empty or size==nqubits). <=0 disables per qubit.
  std::vector<Real> t1;
  std::vector<Real> tphi;

  // Enable controls (global flags).
  bool enable_drive_x = true;
  bool enable_drive_y = false;
  bool enable_drive_z = false;

  // Optional explicit channels (either empty or size==nqubits).
  // ChannelId values must be unique across all control terms.
  std::vector<ChannelId> drive_x;
  std::vector<ChannelId> drive_y;
  std::vector<ChannelId> drive_z;

  // Pair couplings:
  // - exchange_xy: strength * (X_i X_j + Y_i Y_j)
  // - zz:          strength * (Z_i Z_j)
  std::vector<PairCouplingConfig> exchange_xy;
  std::vector<PairCouplingConfig> zz;
};

// Multi-level harmonic oscillator (truncated).
struct OscillatorModelConfig {
  Dim levels = 0;  // truncation dimension (>=2)
  Real omega = Real{0};
  Real tkappa = Real{0};
};

// Kerr oscillator (Duffing-like nonlinearity on oscillator).
struct KerrOscillatorModelConfig {
  Dim levels = 0;  // truncation dimension (>=2)
  Real omega = Real{0};
  Real kerr = Real{0};  // Kerr nonlinearity strength
  Real tkappa = Real{0};
};

// Duffing transmon (multi-level) model.
struct DuffingTransmonModelConfig {
  Dim levels = 0;  // truncation dimension (>=2)
  Real omega = Real{0};
  Real anharm = Real{0};
  Real drive_scale = Real{1};
  Real t1 = Real{0};
  Real tphi = Real{0};
};

// -------------------- Top-level config --------------------

struct Config {
  // Selects which model block is active.
  // Suggested values:
  // - "generic_qubit"
  // - "coupled_qubits"
  // - "oscillator"
  // - "kerr_oscillator"
  // - "duffing_transmon"
  std::string model_type;

  // Run/output/trace options.
  RunOptionsConfig run;
  OutputConfig output;
  TraceConfig trace;

  // Model parameter blocks (only one used depending on model_type).
  GenericQubitModelConfig generic_qubit;
  CoupledQubitsModelConfig coupled_qubits;
  OscillatorModelConfig oscillator;
  KerrOscillatorModelConfig kerr_oscillator;
  DuffingTransmonModelConfig duffing_transmon;

  // Classical controls.
  PulseProgram program;

  // Initial state and observables.
  InitialStateConfig initial_state;
  ObservableConfig observables;
};

}  // namespace fluxpulse
