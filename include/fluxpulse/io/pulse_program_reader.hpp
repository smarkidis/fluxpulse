// include/fluxpulse/io/pulse_program_reader.hpp
#pragma once

#include <cstddef>
#include <string>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/io/config_types.hpp"

namespace fluxpulse {

// PulseProgram is a plain struct (see control/pulse_program.hpp).
struct PulseProgram;

// Load a Phase-1 PulseProgram JSON.
//
// Accepts either:
//   (A) a full Phase-1 config file containing {"program": {...}}, or
//   (B) a standalone program object {"grid": {...}, "pulses": [...]}.
//
// On parse error, if `error_pos` is non-null, it may be filled with a byte offset
// (as returned by the JSON reader) to help locate the issue.
Status load_pulse_program_json_file(const std::string& path,
                                   PulseProgram* out,
                                   size_t* error_pos = nullptr);

// Load a Phase-2 compiled program JSON (portable primitives) and convert it to
// a Phase-1 PulseProgram (channels + segments on a uniform time grid).
//
// Supported v1 primitives (COMPILED_PROGRAM_SCHEMA.md):
//   - rotate (xy-plane axis, angle, duration, shape)
//   - zz_phase (phase between qubit pair -> drift-only idle duration)
//   - wait
//   - frame_z (virtual frame update)
//   - barrier (optional dt alignment)
//
// Conversion uses Phase-1 model conventions. Currently implemented for
// `Config.model_type == "coupled_qubits"`.
//
// Notes:
// - Unknown JSON fields are ignored.
// - Unknown instruction `op` is an error.
Status load_compiled_program_json_file(const std::string& path,
                                      const Config& cfg,
                                      PulseProgram* out,
                                      size_t* error_pos = nullptr);

}  // namespace fluxpulse
