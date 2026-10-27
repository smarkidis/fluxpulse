# FluxPulse Phase 1 Runner

This repo includes a small CLI app:

- `apps/fluxpulse_run.cpp` → builds the `fluxpulse_run` executable

It loads a JSON config, runs a simulation (Phase 1: MCWF trajectories), and writes outputs:
- a results file (JSON or CSV)
- an optional Perfetto-compatible trace file (Chrome JSON trace format)

---

## 1) Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Depending on your CMake setup, the executable will typically be at:

- `build/fluxpulse_run`

If your project builds into a subfolder (e.g. `build/apps/fluxpulse_run`), use the path printed by your build system.

---

## 2) Run with a config

Run the CLI by passing a config JSON path:

```bash
./build/fluxpulse_run configs/1q_rabi.json
```

If successful, it prints:

```
OK
```

and writes output files as specified in the config, for example:

- `out/1q_rabi_result.json`
- `out/1q_rabi_trace.json`

> Note: make sure the `out/` directory exists, or change the paths in the config to an existing directory.

---

## 3) Included configs (Phase 1)

The repository includes example configs under `configs/`:

- `configs/1q_rabi.json`
- `configs/1q_ramsey.json`
- `configs/1q_t1.json`

They cover:
- **Rabi** oscillations (drive on X channel)
- **Ramsey** sequence (two pulses separated by idle + detuning)
- **T1** decay (amplitude damping noise, start in excited state)

---

## 4) Output formats

### A) Result file

`output.result_path` controls the result filename.

- If the filename ends with `.csv`, FluxPulse writes CSV.
- Otherwise it writes JSON.

**CSV format**
- header: `t,<obs0>,<obs1>,...`
- one row per time sample

**JSON format**
- includes:
  - `tlist`
  - `observables`
  - `data` array with `{name, values}`
  - optionally an `expectations` object if observable names are unique and non-empty

### B) Trace file (Perfetto)

If:

```json
"output": { "write_trace": true, "trace_path": "out/trace.json" }
```

then FluxPulse writes a Chrome JSON trace, which can be loaded in Perfetto UI.

The trace shows:
- per-channel tracks (`tid = channel.value`)
- slice events for segments (if enabled)
- optional counter events for sampled amplitudes (if enabled)

---

## 5) Viewing the trace in Perfetto

1) Open the Perfetto UI in a browser (search “Perfetto UI”)  
2) Drag-and-drop the `out/*.json` trace file into the UI

or, if you have a local `perfetto` tool installed, you can open via your system’s method for launching Perfetto UI and load the file.

**Config knobs**
Inside the config, the `trace` section can control emission:

```json
"trace": {
  "emit_segments": true,
  "emit_counters": true,
  "counter_stride": 10,
  "time_scale_to_us": 1.0
}
```

- `emit_segments`: slice events per envelope segment
- `emit_counters`: counter events for sampled coefficients
- `counter_stride`: sample every N grid points
- `time_scale_to_us`: multiplies simulation time to convert to microseconds for the trace
  - use `1.0` if your simulation time is already in microseconds
  - use `1e6` if your simulation time is in seconds

---

## 6) Common issues

### “Failed to open output file for writing”
Ensure the output directory exists, e.g.:

```bash
mkdir -p out
```

### “config schema/validation failed”
Typical causes:
- missing required fields like `model.type` or `program.grid.dt`
- envelope segments overlapping
- duplicate pulses on the same channel (Phase 1 convention)

### Slow runtime on T1
`configs/1q_t1.json` runs many trajectories (e.g. 2000) to reduce Monte Carlo noise.
Reduce `run.trajectories` to speed up at the cost of noisier curves.

---

## 7) Minimal example config

The smallest valid config is roughly:

```json
{
  "model": { "type": "generic_qubit", "omega": 0.0 },
  "program": { "grid": { "dt": 0.01, "n_steps": 101 } },
  "observables": ["Z"],
  "output": { "result_path": "out/result.json" }
}
```

---

## Summary

- Build with CMake
- Run: `./build/fluxpulse_run <config.json>`
- Inspect result JSON/CSV
- Load the trace JSON into Perfetto UI to visualize pulse timing
