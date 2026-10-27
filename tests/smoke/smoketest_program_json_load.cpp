// tests/smoke/smoketest_program_json_load.cpp
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "fluxpulse/control/channel_id.hpp"
#include "fluxpulse/control/pulse_program.hpp"
#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/io/config_types.hpp"
#include "fluxpulse/io/pulse_program_reader.hpp"

namespace {

std::string tmp_file_name(const char* stem) {
#if defined(__unix__) || defined(__APPLE__)
  const int pid = static_cast<int>(::getpid());
  return std::string(stem) + "_" + std::to_string(pid) + ".json";
#else
  return std::string(stem) + ".json";
#endif
}

void write_text_file(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  assert(f.good());
  f << text;
  f.flush();
  assert(f.good());
}

}  // namespace

int main() {
  using fluxpulse::ChannelId;
  using fluxpulse::Config;
  using fluxpulse::Dim;
  using fluxpulse::PulseProgram;
  using fluxpulse::Real;
  using fluxpulse::Status;

  // ---------------------------------------------------------------------------
  // 1) Smoke: Load a standalone Phase-1 program JSON (grid + pulses).
  // ---------------------------------------------------------------------------
  const std::string phase1_prog_path = tmp_file_name("_tmp_phase1_program");
  const std::string phase1_prog_json = R"JSON(
{
  "grid": { "t0": 0.0, "dt": 0.001, "n_points": 11 },
  "pulses": [
    {
      "channel": "drive_x0",
      "envelope": {
        "type": "piecewise_constant",
        "segments": [
          { "t0": 0.0, "t1": 0.004, "value": 0.0 },
          { "t0": 0.004, "t1": 0.008, "value": 1.0 },
          { "t0": 0.008, "t1": 0.010, "value": 0.0 }
        ]
      }
    }
  ]
}
)JSON";
  write_text_file(phase1_prog_path, phase1_prog_json);

  PulseProgram p1;
  size_t err_pos = 0;
  Status st = fluxpulse::load_pulse_program_json_file(phase1_prog_path, &p1, &err_pos);
  assert(st.ok());

  assert(p1.grid.is_uniform());
  assert(p1.grid.dt == Real{0.001});
  assert(p1.grid.size() == Dim{11});
  assert(p1.num_pulses() == Dim{1});
  assert(p1.find_pulse_index(ChannelId{1}) >= 0);  // drive_x0 convention: 1 + 3*q0 = 1

  // Ensure coefficient is as expected on the grid.
  {
    PulseProgram::Cursor cur = p1.make_cursor();
    // At t=0.005 (i=5), segment should be "1.0".
    const Real u = p1.coeff_on_grid(ChannelId{1}, Dim{5}, &cur);
    assert(u > Real{0.5});
  }

  // ---------------------------------------------------------------------------
  // 2) Smoke: Load a Phase-2 compiled program JSON (Bell prep via CZ / ZZ phase)
  //           and convert it to a Phase-1 PulseProgram.
  //
  // This checks the Phase-2 -> Phase-1 integration surface:
  //   compiled instructions -> sampled pulses + idle (ZZ phase -> duration)
  // ---------------------------------------------------------------------------
  const std::string compiled_path = tmp_file_name("_tmp_compiled_bell");
  const std::string compiled_json = R"JSON(
{
  "version": 1,
  "time_unit": "us",
  "schedule": { "type": "asap" },
  "grid_hint": { "dt": 0.001, "align_to_dt": true, "rounding": "nearest" },
  "instructions": [
    { "op": "rotate", "qubit": 0, "axis": { "plane": "xy", "phase": 0.0 }, "angle": 3.141592653589793, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } },
    { "op": "rotate", "qubit": 0, "axis": { "plane": "xy", "phase": 1.5707963267948966 }, "angle": 1.5707963267948966, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } },

    { "op": "rotate", "qubit": 1, "axis": { "plane": "xy", "phase": 0.0 }, "angle": 3.141592653589793, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } },
    { "op": "rotate", "qubit": 1, "axis": { "plane": "xy", "phase": 1.5707963267948966 }, "angle": 1.5707963267948966, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } },

    { "op": "zz_phase", "qubits": [0, 1], "phase": 1.5707963267948966, "frame": "ZZ_over_2" },
    { "op": "frame_z", "qubit": 0, "phase": -1.5707963267948966 },
    { "op": "frame_z", "qubit": 1, "phase": -1.5707963267948966 },

    { "op": "rotate", "qubit": 1, "axis": { "plane": "xy", "phase": 0.0 }, "angle": 3.141592653589793, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } },
    { "op": "rotate", "qubit": 1, "axis": { "plane": "xy", "phase": 1.5707963267948966 }, "angle": 1.5707963267948966, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } },

    { "op": "rotate", "qubit": 1, "axis": { "plane": "xy", "phase": 0.0 }, "angle": 3.141592653589793, "duration": 0.02,
      "shape": { "type": "gaussian", "sigma": 0.006, "truncate": 3.0 } }
  ]
}
)JSON";
  write_text_file(compiled_path, compiled_json);

  // Minimal Phase-1 config needed by the converter: coupled_qubits + drives + ZZ strength.
  Config cfg;
  cfg.model_type = "coupled_qubits";
  cfg.coupled_qubits.nqubits = 2;
  cfg.coupled_qubits.enable_drive_x = true;
  cfg.coupled_qubits.enable_drive_y = true;
  cfg.coupled_qubits.enable_drive_z = false;

  // Ensure zz_phase conversion is possible.
  fluxpulse::PairCouplingConfig zz01;
  zz01.q0 = 0;
  zz01.q1 = 1;
  zz01.strength = Real{0.5};
  cfg.coupled_qubits.zz.push_back(zz01);

  PulseProgram p2;
  size_t err_pos2 = 0;
  st = fluxpulse::load_compiled_program_json_file(compiled_path, cfg, &p2, &err_pos2);
  assert(st.ok());

  assert(p2.grid.is_uniform());
  assert(p2.grid.dt == Real{0.001});
  assert(p2.grid.size() > Dim{100});  // should be ~1700 points for this Bell schedule
  assert(p2.num_pulses() >= Dim{2});

  // Check that expected channels exist (convention drive_x0=1, drive_y0=2, drive_x1=4, drive_y1=5).
  assert(p2.find_pulse_index(ChannelId{1}) >= 0);
  assert(p2.find_pulse_index(ChannelId{2}) >= 0);
  assert(p2.find_pulse_index(ChannelId{4}) >= 0);
  assert(p2.find_pulse_index(ChannelId{5}) >= 0);

  // Validate again for good measure (should be idempotent).
  st = p2.validate(/*finalize_envelopes=*/true);
  assert(st.ok());

  return 0;
}
