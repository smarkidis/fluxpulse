// include/fluxpulse/io/config_loader.hpp
#pragma once

#include <string>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/io/config_types.hpp"
#include "fluxpulse/io/json_reader.hpp"

namespace fluxpulse {

// Load a configuration file from disk and populate a Config struct.
//
// Responsibilities (Phase 1):
// - Read JSON from `path`
// - Validate top-level structure and required fields
// - Build:
//   - cfg->program (TimeGrid + Pulses + Envelopes)
//   - cfg->run (dt, t_final, trajectories, base_seed, sample_stride, renormalize_each_step)
//   - cfg->output / cfg->trace
//   - cfg->model_type + model param block
//     - Phase 1 supports: generic_qubit, coupled_qubits
//   - cfg->initial_state (optional; defaults to all-zero basis state)
//   - cfg->observables
// - Call validate/finalize for:
//   - TimeGrid
//   - Envelope(s)
//   - PulseProgram
//
// Notes:
// - No exceptions. Returns Status with ErrorCode::kIo or ErrorCode::kParse for file/parse errors,
//   ErrorCode::kInvalidArgument for schema/validation errors.
// - Phase 1 supports only one Pulse per channel (PulseProgram validates this).
//
// Optional output:
// - error_pos: byte offset in the JSON text (only meaningful for parse errors).
Status load_config_file(const std::string& path, Config* cfg, size_t* error_pos);

// Load from already-parsed JsonValue. Useful for tests.
Status load_config_from_json(const JsonValue& root, Config* cfg);

}  // namespace fluxpulse
