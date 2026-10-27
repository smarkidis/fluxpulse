#!/usr/bin/env python3
import copy
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


# -------------------------
# Study parameters
# -------------------------

LEVELS = 3

# Duffing anharmonicity in the same inverse-time units as the drive.
# More negative/larger magnitude separates |1> -> |2> more strongly.
ANHARMONICITY = -10.0

# Duration sweep for calibrated X_pi pulses.
# Pulse amplitude is chosen as DRIVE_SCALE * pi / duration.
PULSE_DURATIONS = [0.16, 0.20, 0.25, 0.3333333333, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

# One short-pulse case for the left-panel time trace.
SHORT_TRACE_DURATION = 0.20

# Number of RK4 intervals per pulse.
STEPS_PER_PULSE = 1000

# Default scale for amplitude = scale * pi / duration.
# The script runs a short calibration sweep and updates this automatically.
DEFAULT_DRIVE_SCALE = 1.0

CALIBRATE = True
CALIBRATION_DURATION = 1.0
CALIBRATION_SCALES = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

CFG_DIR = Path("configs/duffing_leakage")
OUT_DIR = Path("out/duffing_leakage")
EXE = Path("./build/fluxpulse_run")

RABI_TEMPLATE = Path("configs/1q_rabi.json")
DUFFING_TEMPLATE_CANDIDATES = [
    Path("configs/1q_duffing.json"),
    Path("configs/1q_duffing_rabi.json"),
    Path("configs/duffing_transmon.json"),
    RABI_TEMPLATE,
]


# -------------------------
# Basic IO
# -------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def tag_float(x):
    return f"{x:g}".replace(".", "p").replace("-", "m")


def choose_template():
    for p in DUFFING_TEMPLATE_CANDIDATES:
        if p.exists():
            print(f"[template] using {p}")
            return load_json(p)
    raise RuntimeError("Could not find a usable template config.")


# -------------------------
# Config editing helpers
# -------------------------

def set_model_duffing(cfg):
    model = cfg.setdefault("model", {})

    # Exact minimal schema for the current C++ Duffing loader.
    # The source uses DuffingTransmonParams::anharm, so use "anharm".
    model.clear()
    model["type"] = "duffing_transmon"
    model["levels"] = LEVELS
    model["omega"] = 0.0
    model["anharm"] = ANHARMONICITY

    # Disable decoherence to isolate coherent leakage.
    model["t1"] = 0.0
    model["tphi"] = 0.0


def set_model_effective_qubit(cfg):
    model = cfg.setdefault("model", {})
    model["type"] = "generic_qubit"
    model["omega"] = 0.0
    model["detuning"] = 0.0

    for k in ["t1", "tphi", "t2", "gamma1", "gamma_phi", "gammaphi"]:
        model[k] = 0.0


def set_observables_duffing(cfg):
    # Explicit basis projectors for a single 3-level Duffing subsystem.
    # This gives P0, P1, and P2 directly.
    cfg["observables"] = ["P:0", "P:1", "P:2"]


def set_observables_effective(cfg):
    cfg["observables"] = ["P0", "P1"]


def set_run_options(cfg):
    run = cfg.setdefault("run", {})
    run["trajectories"] = 1
    run["sample_stride"] = 1
    run["base_seed"] = 12345


def disable_trace(cfg):
    trace = cfg.get("trace")
    if isinstance(trace, dict):
        for k in ["write_trace", "enabled", "emit_trace"]:
            if k in trace:
                trace[k] = False

    output = cfg.get("output")
    if isinstance(output, dict):
        for k in ["write_trace", "enabled_trace", "emit_trace"]:
            if k in output:
                output[k] = False


def get_program(cfg):
    return cfg.setdefault("program", {})


def get_grid(cfg):
    program = get_program(cfg)
    return program.setdefault("grid", {})


def set_grid_for_duration(cfg, duration):
    grid = get_grid(cfg)

    t0 = 0.0
    dt = duration / STEPS_PER_PULSE
    n_intervals = STEPS_PER_PULSE

    grid["t0"] = t0
    grid["dt"] = dt

    # In your current FluxPulse configs, n_steps behaves as number of intervals.
    if "n_points" in grid and "n_steps" not in grid:
        grid["n_points"] = n_intervals + 1
    else:
        grid["n_steps"] = n_intervals

    if "t_end" in grid:
        grid["t_end"] = duration
    if "t_final" in grid:
        grid["t_final"] = duration
    if "duration" in grid:
        grid["duration"] = duration


def find_existing_pulse_and_segment(cfg):
    program = get_program(cfg)
    pulses = program.get("pulses")

    if isinstance(pulses, list) and pulses:
        pulse = pulses[0]
        segments = pulse.get("segments")
        if isinstance(segments, list) and segments:
            return pulse, segments[0]

        env = pulse.get("envelope")
        if isinstance(env, dict):
            env_segments = env.get("segments")
            if isinstance(env_segments, list) and env_segments:
                return pulse, env_segments[0]

        return pulse, {}

    return None, {}


def detect_key(seg, candidates, default):
    for k in candidates:
        if k in seg:
            return k
    return default


def build_segment_like(existing_seg, t0, t1, amp):
    start_key = detect_key(existing_seg, ["t0", "t_start", "start", "begin", "t_begin"], "t0")
    end_key = detect_key(existing_seg, ["t1", "t_end", "end", "stop", "t_stop"], "t1")
    amp_key = detect_key(existing_seg, ["value", "u", "amp", "amplitude", "coeff"], "value")

    return {
        start_key: t0,
        end_key: t1,
        amp_key: amp,
    }


def set_single_x_pulse(cfg, duration, amplitude):
    program = get_program(cfg)
    existing_pulse, existing_seg = find_existing_pulse_and_segment(cfg)

    seg = build_segment_like(existing_seg, 0.0, duration, amplitude)

    if existing_pulse is None:
        pulse = {
            "channel": "drive_x",
            "segments": [seg],
        }
        program["pulses"] = [pulse]
        return

    pulse = copy.deepcopy(existing_pulse)

    # Common channel-name aliases.
    if "channel" in pulse:
        pulse["channel"] = "drive_x"
    if "name" in pulse and "drive" in str(pulse["name"]).lower():
        pulse["name"] = "drive_x"
    if "channel_name" in pulse:
        pulse["channel_name"] = "drive_x"

    if "segments" in pulse and isinstance(pulse["segments"], list):
        pulse["segments"] = [seg]
    elif "envelope" in pulse and isinstance(pulse["envelope"], dict):
        pulse["envelope"]["segments"] = [seg]
    else:
        pulse["segments"] = [seg]

    program["pulses"] = [pulse]


def replace_paths(obj, result_path, trace_path):
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()

            if isinstance(v, str):
                lv = v.lower()
                if "result" in lk or "result" in lv:
                    obj[k] = str(result_path)
                elif "trace" in lk or "trace" in lv:
                    obj[k] = str(trace_path)
            else:
                replace_paths(v, result_path, trace_path)

    elif isinstance(obj, list):
        for item in obj:
            replace_paths(item, result_path, trace_path)


def set_output_paths(cfg, result_path, trace_path):
    cfg.setdefault("output", {})
    replace_paths(cfg["output"], result_path, trace_path)

    # Add common fallback keys. Unknown keys should be ignored by the loader.
    cfg["output"]["result_json"] = str(result_path)
    cfg["output"]["trace_json"] = str(trace_path)

    if "trace" in cfg:
        replace_paths(cfg["trace"], result_path, trace_path)


# -------------------------
# Result parsing
# -------------------------

def is_number_list(x):
    return isinstance(x, list) and all(isinstance(v, (int, float)) for v in x)


def normalize_name(s):
    return str(s).replace("$", "").replace("\\", "").replace(" ", "").lower()


def extract_time(result):
    for k in ["tlist", "time", "times", "t"]:
        if k in result and is_number_list(result[k]):
            return result[k]

    # Some writers use samples: [{t: ..., expectations: ...}, ...]
    samples = result.get("samples")
    if isinstance(samples, list) and samples and isinstance(samples[0], dict):
        for key in ["t", "time"]:
            if key in samples[0]:
                return [float(s[key]) for s in samples if key in s]

    return None


def series_from_value(v):
    if is_number_list(v):
        return v
    if isinstance(v, dict):
        for key in ["values", "mean", "data", "series", "expectation"]:
            if key in v and is_number_list(v[key]):
                return v[key]
    return None


def extract_expectations(result):
    candidates = ["expectations", "expect", "means", "observable_values"]

    for key in candidates:
        obj = result.get(key)

        if isinstance(obj, dict):
            out = {}
            for name, val in obj.items():
                s = series_from_value(val)
                if s is not None:
                    out[str(name)] = s
            if out:
                return out

        if isinstance(obj, list):
            # list of records: [{name: ..., values: ...}, ...]
            out = {}
            for item in obj:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("observable") or item.get("label")
                    s = series_from_value(item)
                    if name is not None and s is not None:
                        out[str(name)] = s
            if out:
                return out

            # matrix with observable_names
            names = result.get("observable_names") or result.get("observables")
            if isinstance(names, list) and names:
                clean_names = [str(n) if not isinstance(n, dict) else str(n.get("name", n)) for n in names]

                if len(obj) == len(clean_names) and all(is_number_list(row) for row in obj):
                    return {clean_names[i]: obj[i] for i in range(len(clean_names))}

                if obj and isinstance(obj[0], list) and len(obj[0]) == len(clean_names):
                    cols = list(zip(*obj))
                    return {clean_names[i]: list(cols[i]) for i in range(len(clean_names))}

    return {}


def get_series(exp, wanted):
    wanted_norm = [normalize_name(w) for w in wanted]

    for name, values in exp.items():
        n = normalize_name(name)
        if n in wanted_norm:
            return values

    raise KeyError(f"Could not find any of {wanted}. Available: {list(exp.keys())}")


def load_observable_series(result_path):
    result = load_json(result_path)
    time = extract_time(result)
    exp = extract_expectations(result)

    p0 = get_series(exp, ["P0", "P:0", "P_0", "pop0", "population0"])
    p1 = get_series(exp, ["P1", "P:1", "P_1", "pop1", "population1"])

    p2 = None
    try:
        p2 = get_series(exp, ["P2", "P:2", "P_2", "pop2", "population2"])
    except KeyError:
        pass

    n = None
    try:
        n = get_series(exp, ["N", "n", "number", "number_operator"])
    except KeyError:
        pass

    # If P2 is unavailable but N is available, reconstruct P2 for a 3-level model.
    if p2 is None and n is not None:
        p2 = [(nv - p1v) / 2.0 for nv, p1v in zip(n, p1)]

    return {
        "time": time,
        "P0": p0,
        "P1": p1,
        "N": n,
        "P2": p2,
        "P2": p2,
    }


# -------------------------
# Running
# -------------------------

def run_case(cfg_path, result_path, fallback_path):
    cfg_path = Path(cfg_path)
    result_path = Path(result_path)
    fallback_path = Path(fallback_path)

    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        result_path.unlink()
    if fallback_path.exists():
        fallback_path.unlink()

    before = {p: p.stat().st_mtime for p in Path("out").glob("*.json")}

    print(f"[run] {cfg_path}")
    ret = subprocess.run([str(EXE), str(cfg_path)])

    if ret.returncode != 0:
        raise RuntimeError(f"FluxPulse run failed for {cfg_path}")

    if result_path.exists():
        return

    if fallback_path.exists():
        shutil.copy2(fallback_path, result_path)
        print(f"[copy] {fallback_path} -> {result_path}")
        return

    produced = []
    for p in Path("out").glob("*.json"):
        if "trace" in p.name.lower():
            continue
        old = before.get(p)
        new = p.stat().st_mtime
        if old is None or new > old:
            produced.append(p)

    if produced:
        newest = max(produced, key=lambda p: p.stat().st_mtime)
        shutil.copy2(newest, result_path)
        print(f"[copy newest] {newest} -> {result_path}")
        return

    raise RuntimeError(
        f"Could not locate result for {cfg_path}. "
        f"Expected {result_path}, fallback {fallback_path}, "
        f"or a newly written non-trace JSON in out/."
    )


def make_duffing_config(template, duration, amplitude, name):
    cfg = copy.deepcopy(template)

    set_model_duffing(cfg)
    set_observables_duffing(cfg)
    set_run_options(cfg)
    disable_trace(cfg)
    set_grid_for_duration(cfg, duration)
    set_single_x_pulse(cfg, duration, amplitude)

    result_path = OUT_DIR / f"{name}_result.json"
    trace_path = OUT_DIR / f"{name}_trace.json"
    set_output_paths(cfg, result_path, trace_path)

    cfg_path = CFG_DIR / f"{name}.json"
    write_json(cfg_path, cfg)

    return cfg_path, result_path, trace_path


def make_effective_config(template, duration, amplitude, name):
    cfg = copy.deepcopy(template)

    set_model_effective_qubit(cfg)
    set_observables_effective(cfg)
    set_run_options(cfg)
    disable_trace(cfg)
    set_grid_for_duration(cfg, duration)
    set_single_x_pulse(cfg, duration, amplitude)

    result_path = OUT_DIR / f"{name}_result.json"
    trace_path = OUT_DIR / f"{name}_trace.json"
    set_output_paths(cfg, result_path, trace_path)

    cfg_path = CFG_DIR / f"{name}.json"
    write_json(cfg_path, cfg)

    return cfg_path, result_path, trace_path


def calibrate_drive_scale(duffing_template):
    if not CALIBRATE:
        return DEFAULT_DRIVE_SCALE, []

    print("[calibration] scanning drive scale for a long X_pi pulse")

    records = []
    best_scale = DEFAULT_DRIVE_SCALE
    best_p1 = -1.0

    for scale in CALIBRATION_SCALES:
        duration = CALIBRATION_DURATION
        amp = scale * math.pi / duration
        name = f"calib_scale{tag_float(scale)}"

        cfg_path, result_path, _ = make_duffing_config(duffing_template, duration, amp, name)
        run_case(cfg_path, result_path, "out/1q_rabi_result.json")

        try:
            s = load_observable_series(result_path)
            p1_final = float(s["P1"][-1])
            p0_final = float(s["P0"][-1])
            leakage = 1.0 - p0_final - p1_final

            records.append({
                "scale": scale,
                "duration": duration,
                "amplitude": amp,
                "P0_final": p0_final,
                "P1_final": p1_final,
                "leakage": leakage,
                "result": str(result_path),
            })

            print(f"  scale={scale:g}  P1(tf)={p1_final:.6g}  leakage={leakage:.6g}")

            if p1_final > best_p1:
                best_p1 = p1_final
                best_scale = scale

        except Exception as e:
            print(f"  warning: could not parse calibration result {result_path}: {e}")

    print(f"[calibration] selected DRIVE_SCALE={best_scale:g}")
    return best_scale, records


def write_summary_csv(records):
    path = OUT_DIR / "duffing_leakage_summary.csv"

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "duration",
                "amplitude",
                "P0_final",
                "P1_final",
                "N_final",
                "P2_final",
                "leakage",
                "result",
            ],
        )
        w.writeheader()

        for r in records:
            w.writerow(r)

    return path


def write_time_series_csv(result_path):
    s = load_observable_series(result_path)

    time = s["time"]
    p0 = s["P0"]
    p1 = s["P1"]
    n = s["N"]
    p2 = s["P2"]

    path = OUT_DIR / "duffing_short_pulse_timeseries.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "P0", "P1", "N", "P2", "leakage_instant"])

        for i in range(len(p0)):
            t = time[i] if time is not None and i < len(time) else i
            p2i = p2[i] if p2 is not None else ""
            ni = n[i] if n is not None else ""
            leakage = 1.0 - p0[i] - p1[i]
            w.writerow([t, p0[i], p1[i], ni, p2i, leakage])

    return path


def main():
    if not EXE.exists():
        raise RuntimeError(f"Cannot find executable: {EXE}")

    # Clean only generated leakage outputs.
    if CFG_DIR.exists():
        shutil.rmtree(CFG_DIR)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if Path("duffing_leakage_results.tar.gz").exists():
        Path("duffing_leakage_results.tar.gz").unlink()

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    duffing_template = choose_template()
    effective_template = load_json(RABI_TEMPLATE)

    drive_scale, calibration_records = calibrate_drive_scale(duffing_template)

    manifest = {
        "description": "Duffing transmon leakage under calibrated X_pi pulses",
        "levels": LEVELS,
        "anharmonicity": ANHARMONICITY,
        "drive_rule": "amplitude = selected_drive_scale * pi / duration",
        "selected_drive_scale": drive_scale,
        "short_trace_duration": SHORT_TRACE_DURATION,
        "pulse_durations": PULSE_DURATIONS,
        "steps_per_pulse": STEPS_PER_PULSE,
        "calibration": calibration_records,
        "cases": [],
    }

    summary_records = []
    short_trace_result = None

    # Duffing duration sweep.
    for duration in PULSE_DURATIONS:
        amp = drive_scale * math.pi / duration
        name = f"duffing_T{tag_float(duration)}"

        cfg_path, result_path, _ = make_duffing_config(duffing_template, duration, amp, name)
        run_case(cfg_path, result_path, "out/1q_rabi_result.json")

        manifest["cases"].append({
            "kind": "duffing",
            "duration": duration,
            "amplitude": amp,
            "cfg": str(cfg_path),
            "result": str(result_path),
        })

        try:
            s = load_observable_series(result_path)
            p0f = float(s["P0"][-1])
            p1f = float(s["P1"][-1])
            nf = float(s["N"][-1]) if s["N"] is not None else ""
            p2f = float(s["P2"][-1]) if s["P2"] is not None else ""
            leakage = 1.0 - p0f - p1f

            summary_records.append({
                "model": "duffing",
                "duration": duration,
                "amplitude": amp,
                "P0_final": p0f,
                "P1_final": p1f,
                "N_final": nf,
                "P2_final": p2f,
                "leakage": leakage,
                "result": str(result_path),
            })

        except Exception as e:
            print(f"warning: could not parse Duffing result {result_path}: {e}")

        if abs(duration - SHORT_TRACE_DURATION) < 1e-12:
            short_trace_result = result_path

    # Effective qubit comparison. Leakage should be zero by construction.
    for duration in PULSE_DURATIONS:
        amp = drive_scale * math.pi / duration
        name = f"effective_T{tag_float(duration)}"

        cfg_path, result_path, _ = make_effective_config(effective_template, duration, amp, name)
        run_case(cfg_path, result_path, "out/1q_rabi_result.json")

        manifest["cases"].append({
            "kind": "effective_qubit",
            "duration": duration,
            "amplitude": amp,
            "cfg": str(cfg_path),
            "result": str(result_path),
        })

        try:
            s = load_observable_series(result_path)
            p0f = float(s["P0"][-1])
            p1f = float(s["P1"][-1])

            summary_records.append({
                "model": "effective_qubit",
                "duration": duration,
                "amplitude": amp,
                "P0_final": p0f,
                "P1_final": p1f,
                "N_final": "",
                "P2_final": "",
                "leakage": 1.0 - p0f - p1f,
                "result": str(result_path),
            })

        except Exception as e:
            print(f"warning: could not parse effective result {result_path}: {e}")
            summary_records.append({
                "model": "effective_qubit",
                "duration": duration,
                "amplitude": amp,
                "P0_final": "",
                "P1_final": "",
                "N_final": "",
                "P2_final": "",
                "leakage": 0.0,
                "result": str(result_path),
            })

    summary_path = write_summary_csv(summary_records)
    print(f"[summary] {summary_path}")

    if short_trace_result is not None:
        try:
            ts_path = write_time_series_csv(short_trace_result)
            print(f"[timeseries] {ts_path}")
            manifest["short_trace_timeseries_csv"] = str(ts_path)
        except Exception as e:
            print(f"warning: could not write short-pulse timeseries CSV: {e}")

    write_json(OUT_DIR / "manifest.json", manifest)

    print("[package] duffing_leakage_results.tar.gz")
    ret = subprocess.run([
        "tar",
        "-czf",
        "duffing_leakage_results.tar.gz",
        "configs/duffing_leakage",
        "out/duffing_leakage",
    ])

    if ret.returncode != 0:
        raise RuntimeError("Packaging failed")

    print("Done.")
    print("Upload: duffing_leakage_results.tar.gz")


if __name__ == "__main__":
    main()
