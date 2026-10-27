// tests/unit/test_config_loader.cpp
#include "fluxpulse/io/config_loader.hpp"

#include <cassert>
#include <string>

namespace {

inline void expect_ok(const fluxpulse::Status& st) { assert(st.ok()); }
inline void expect_fail(const fluxpulse::Status& st) { assert(!st.ok()); }

inline void expect_near(fluxpulse::Real a, fluxpulse::Real b, fluxpulse::Real eps) {
  const fluxpulse::Real d = (a > b) ? (a - b) : (b - a);
  assert(d <= eps);
}

}  // namespace

int main() {
  using fluxpulse::Config;
  using fluxpulse::JsonValue;
  using fluxpulse::Real;

  constexpr Real kEps = Real{1e-12};

  // ---- Case A: minimal valid config (generic_qubit + uniform grid) ----
  {
    const char* json =
        "{"
        "  \"model\": {\"type\": \"generic_qubit\", \"omega\": 5.0, \"t1\": 25.0},"
        "  \"program\": {"
        "    \"grid\": {\"t0\": 0.0, \"dt\": 0.1, \"n_steps\": 6},"
        "    \"pulses\": ["
        "      {"
        "        \"channel\": \"drive_x\","
        "        \"envelope\": {"
        "          \"type\": \"piecewise_constant\","
        "          \"segments\": ["
        "            {\"t0\": 0.0, \"t1\": 0.3, \"value\": 1.0}"
        "          ]"
        "        }"
        "      }"
        "    ]"
        "  },"
        "  \"observables\": [\"Z\", \"X\"],"
        "  \"output\": {\"write_trace\": true, \"trace_path\": \"out/trace.json\"},"
        "  \"trace\": {\"emit_segments\": true, \"emit_counters\": false}"
        "}";

    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(json, &root, &pos));

    Config cfg;
    expect_ok(fluxpulse::load_config_from_json(root, &cfg));

    // Model
    assert(cfg.model_type == "generic_qubit");
    expect_near(cfg.generic_qubit.omega, Real{5.0}, kEps);
    expect_near(cfg.generic_qubit.t1, Real{25.0}, kEps);

    // Program/grid
    assert(cfg.program.grid.is_uniform());
    expect_ok(cfg.program.grid.validate());
    assert(cfg.program.grid.size() == 6);
    expect_near(cfg.program.grid.t_end(), Real{0.5}, kEps);  // t0 + dt*(n-1)

    // Run defaults derived from grid
    expect_near(cfg.run.dt, Real{0.1}, kEps);
    expect_near(cfg.run.t_final, Real{0.5}, kEps);
    assert(cfg.run.trajectories == 1u);

    // Output/trace
    assert(cfg.output.write_trace == true);
    assert(cfg.output.trace_path == "out/trace.json");
    assert(cfg.trace.emit_segments == true);
    assert(cfg.trace.emit_counters == false);

    // Observables
    assert(cfg.observables.names.size() == 2);
    assert(cfg.observables.names[0] == "Z");
    assert(cfg.observables.names[1] == "X");

    // Envelope should be finalized by program.validate(true) inside loader.
    // Coefficient checks:
    expect_near(cfg.program.coeff(fluxpulse::ChannelId::DriveX(), Real{0.1}), Real{1.0}, kEps);
    // half-open boundary: [0,0.3) excludes t=0.3
    expect_near(cfg.program.coeff(fluxpulse::ChannelId::DriveX(), Real{0.3}), Real{0.0}, kEps);
  }

  // ---- Case B: explicit grid + oscillator model + observables object form ----
  {
    const char* json =
        "{"
        "  \"model\": {\"type\": \"oscillator\", \"levels\": 3, \"omega\": 2.0},"
        "  \"program\": {"
        "    \"grid\": {\"tlist\": [0.0, 0.2, 0.5]},"
        "    \"pulses\": ["
        "      {"
        "        \"channel\": \"drive_z\","
        "        \"envelope\": {"
        "          \"type\": \"piecewise_linear\","
        "          \"segments\": ["
        "            {\"t0\": 0.0, \"t1\": 0.5, \"v0\": 0.0, \"v1\": 1.0}"
        "          ]"
        "        }"
        "      }"
        "    ]"
        "  },"
        "  \"run\": {\"t_final\": 0.5, \"trajectories\": 10, \"base_seed\": 7},"
        "  \"observables\": {\"names\": [\"N\"]}"
        "}";

    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(json, &root, &pos));

    Config cfg;
    expect_ok(fluxpulse::load_config_from_json(root, &cfg));

    assert(cfg.model_type == "oscillator");
    assert(cfg.oscillator.levels == 3);
    expect_near(cfg.oscillator.omega, Real{2.0}, kEps);

    assert(cfg.program.grid.is_explicit());
    expect_ok(cfg.program.grid.validate());
    assert(cfg.program.grid.size() == 3);
    expect_near(cfg.program.grid.t_end(), Real{0.5}, kEps);

    // run.dt derived from first interval for explicit grid
    expect_near(cfg.run.dt, Real{0.2}, kEps);
    expect_near(cfg.run.t_final, Real{0.5}, kEps);
    assert(cfg.run.trajectories == 10u);
    assert(cfg.run.base_seed == 7u);

    assert(cfg.observables.names.size() == 1);
    assert(cfg.observables.names[0] == "N");

    // Linear ramp: u(0.2) = 0.4
    expect_near(cfg.program.coeff(fluxpulse::ChannelId::DriveZ(), Real{0.2}), Real{0.4}, Real{1e-9});
  }

  // ---- Case C: invalid (duplicate channel in Phase 1) ----
  {
    const char* json =
        "{"
        "  \"model\": {\"type\": \"generic_qubit\", \"omega\": 1.0},"
        "  \"program\": {"
        "    \"grid\": {\"dt\": 0.1, \"n_steps\": 3},"
        "    \"pulses\": ["
        "      {"
        "        \"channel\": \"drive_x\","
        "        \"envelope\": {"
        "          \"type\": \"piecewise_constant\","
        "          \"segments\": [{\"t0\": 0.0, \"t1\": 0.1, \"value\": 1.0}]"
        "        }"
        "      },"
        "      {"
        "        \"channel\": \"drive_x\","
        "        \"envelope\": {"
        "          \"type\": \"piecewise_constant\","
        "          \"segments\": [{\"t0\": 0.1, \"t1\": 0.2, \"value\": 2.0}]"
        "        }"
        "      }"
        "    ]"
        "  }"
        "}";

    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(json, &root, &pos));

    Config cfg;
    const fluxpulse::Status st = fluxpulse::load_config_from_json(root, &cfg);
    expect_fail(st);
  }

  // ---- Case D: invalid run.t_final mismatch with grid end time ----
  {
    const char* json =
        "{"
        "  \"model\": {\"type\": \"generic_qubit\", \"omega\": 1.0},"
        "  \"program\": {"
        "    \"grid\": {\"t0\": 0.0, \"dt\": 0.1, \"n_steps\": 6}"
        "  },"
        "  \"run\": {\"t_final\": 1.0}"
        "}";

    JsonValue root;
    size_t pos = 0;
    expect_ok(fluxpulse::parse_json(json, &root, &pos));

    Config cfg;
    const fluxpulse::Status st = fluxpulse::load_config_from_json(root, &cfg);
    expect_fail(st);
  }

  // Optional fields may be omitted, but a supplied value must have the right type.
  for (const char* json : {
      R"({"model":{"type":"generic_qubit","omega":0,"t1":"invalid"},"program":{"grid":{"dt":0.1,"n_points":2}}})",
      R"({"model":{"type":"generic_qubit","omega":0},"program":{"grid":{"dt":0.1,"n_points":2}},"output":{"write_trace":"yes"}})",
      R"({"model":{"type":"generic_qubit","omega":0},"program":{"grid":{"dt":0.1,"n_points":2}},"trace":{"diagnostic_stride":"invalid"}})"}) {
    JsonValue root;
    expect_ok(fluxpulse::parse_json(json, &root, nullptr));
    Config cfg;
    expect_fail(fluxpulse::load_config_from_json(root, &cfg));
  }

  return 0;
}
