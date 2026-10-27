// include/fluxpulse/io/result_writer.hpp
#pragma once

#include <string>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/sim/result.hpp"

namespace fluxpulse {
namespace io {

// Write Result to a JSON file.
//
// Output schema (Phase 1):
// {
//   "tlist": [ ... ],
//   "observables": ["Z0", "X0", ...],           // may be empty strings if unknown
//   "expectations": { "Z0": [ ... ], ... },     // mean; only if names are non-empty and unique
//   "expect_var": { "Z0": [ ... ], ... },       // optional; only if names are non-empty and unique and var is present
//   "expect_stderr": { "Z0": [ ... ], ... },    // optional; only if names are non-empty and unique and stderr is present
//   "data": [                                   // always present; row-major fallback
//     { "name": "...", "values": [ ... ], "var": [ ... ], "stderr": [ ... ] },
//     ...
//   ]
// }
//
// Notes:
// - "expect_var" is variance across trajectories.
// - "expect_stderr" is standard error of the mean: sqrt(var / Ntraj).
Status write_result_json(const std::string& path, const Result& r);

// Write Result to a CSV file.
//
// Format (Phase 1):
// - First column: t
// - Next columns: one per observable mean
// - Optional additional columns if available: <obs>_var and <obs>_stderr
//
// If observable name is empty, it uses "obs0", "obs1", ...
Status write_result_csv(const std::string& path, const Result& r);

}  // namespace io
}  // namespace fluxpulse
