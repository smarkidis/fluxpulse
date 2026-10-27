import copy
import json
import shutil
import subprocess
from pathlib import Path


RK4_STEPS = [0.2, 0.1, 0.05, 0.025, 0.0125]

NTRAJ_VALUES = [25, 50, 100, 200, 400, 800, 1600]
SEEDS = [101, 202, 303, 404, 505]

RK4_TARGET_TOTAL_TIME = 1.0

CFG_DIR = Path("configs/convergence")
OUT_DIR = Path("out/convergence")
EXE = Path("./build/fluxpulse_run")


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


def get_grid(cfg):
    return cfg["program"]["grid"]


def infer_interval(grid):
    t0 = float(grid.get("t0", 0.0))

    if "t_end" in grid:
        return t0, float(grid["t_end"])

    if "t_final" in grid:
        return t0, float(grid["t_final"])

    if "duration" in grid:
        return t0, t0 + float(grid["duration"])

    # FluxPulse n_steps is treated here as the number of time intervals.
    if "dt" in grid and "n_steps" in grid:
        return t0, t0 + float(grid["dt"]) * int(grid["n_steps"])

    if "dt" in grid and "n_points" in grid:
        return t0, t0 + float(grid["dt"]) * (int(grid["n_points"]) - 1)

    raise RuntimeError(f"Cannot infer final time from grid keys: {list(grid.keys())}")


def segment_start_keys():
    return ["t0", "t_start", "start", "begin", "t_begin"]


def segment_end_keys():
    return ["t1", "t_end", "end", "stop", "t_stop"]


def pulse_boundaries(cfg):
    vals = []
    for pulse in cfg.get("program", {}).get("pulses", []):
        for seg in pulse.get("segments", []):
            for k in segment_start_keys() + segment_end_keys():
                if k in seg:
                    vals.append(float(seg[k]))
    return vals


def clip_final_pulse_boundaries(cfg, old_t_end, new_t_end):
    for pulse in cfg.get("program", {}).get("pulses", []):
        for seg in pulse.get("segments", []):
            for k in segment_end_keys():
                if k in seg and abs(float(seg[k]) - old_t_end) < 1e-9:
                    seg[k] = new_t_end


def set_grid_dt(cfg, h, name, target_total_time=None):
    grid = get_grid(cfg)
    t0, old_t_end = infer_interval(grid)

    if target_total_time is None:
        t_end = old_t_end
        total = t_end - t0
    else:
        total = float(target_total_time)
        t_end = t0 + total
        clip_final_pulse_boundaries(cfg, old_t_end, t_end)

    n_intervals = int(round(total / h))
    actual_t_end = t0 + n_intervals * h

    if abs(actual_t_end - t_end) > 1e-10:
        raise RuntimeError(
            f"{name}: h={h} does not divide total time {total}. "
            f"t0={t0}, t_end={t_end}, actual_t_end={actual_t_end}"
        )

    for b in [t0, t_end] + pulse_boundaries(cfg):
        if b < t0 - 1e-9 or b > t_end + 1e-9:
            continue

        q = (b - t0) / h
        if abs(q - round(q)) > 1e-8:
            raise RuntimeError(
                f"{name}: pulse boundary t={b} is not aligned with h={h}. "
                f"(t - t0)/h={q}"
            )

    grid["dt"] = h

    # Important: for FluxPulse, n_steps is the number of intervals.
    if "n_steps" in grid:
        grid["n_steps"] = n_intervals
    elif "n_points" in grid:
        grid["n_points"] = n_intervals + 1
    else:
        grid["n_steps"] = n_intervals

    if "t_end" in grid:
        grid["t_end"] = t_end
    if "t_final" in grid:
        grid["t_final"] = t_end
    if "duration" in grid:
        grid["duration"] = total


def set_run(cfg, key, value):
    cfg.setdefault("run", {})
    cfg["run"][key] = value


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


def zero_noise_for_rabi(cfg):
    model = cfg.get("model", {})
    if not isinstance(model, dict):
        return

    for k in ["t1", "tphi", "t2", "gamma1", "gamma_phi", "gammaphi"]:
        if k in model:
            if isinstance(model[k], list):
                model[k] = [0 for _ in model[k]]
            else:
                model[k] = 0


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

    if "trace" in cfg:
        replace_paths(cfg["trace"], result_path, trace_path)


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


def main():
    if not EXE.exists():
        raise RuntimeError(f"Cannot find executable: {EXE}")

    # Clean only generated convergence outputs.
    if CFG_DIR.exists():
        shutil.rmtree(CFG_DIR)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if Path("convergence_results.tar.gz").exists():
        Path("convergence_results.tar.gz").unlink()

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rabi = load_json("configs/1q_rabi.json")
    t1 = load_json("configs/1q_t1.json")

    manifest = {
        "description": "FluxPulse numerical convergence study",
        "rk4_target_total_time": RK4_TARGET_TOTAL_TIME,
        "rk4_steps": RK4_STEPS,
        "ntraj_values": NTRAJ_VALUES,
        "seeds": SEEDS,
        "cases": [],
    }

    for h in RK4_STEPS:
        cfg = copy.deepcopy(rabi)
        name = f"rk4_h{tag_float(h)}"

        cfg_path = CFG_DIR / f"{name}.json"
        result_path = OUT_DIR / f"{name}_result.json"
        trace_path = OUT_DIR / f"{name}_trace.json"

        set_grid_dt(cfg, h, name, target_total_time=RK4_TARGET_TOTAL_TIME)
        set_run(cfg, "sample_stride", 1)
        set_run(cfg, "trajectories", 1)
        zero_noise_for_rabi(cfg)
        disable_trace(cfg)
        set_output_paths(cfg, result_path, trace_path)

        write_json(cfg_path, cfg)

        manifest["cases"].append({
            "kind": "rk4",
            "h": h,
            "cfg": str(cfg_path),
            "result": str(result_path),
        })

        run_case(cfg_path, result_path, "out/1q_rabi_result.json")

    for ntraj in NTRAJ_VALUES:
        for seed in SEEDS:
            cfg = copy.deepcopy(t1)
            name = f"mcwf_N{ntraj}_seed{seed}"

            cfg_path = CFG_DIR / f"{name}.json"
            result_path = OUT_DIR / f"{name}_result.json"
            trace_path = OUT_DIR / f"{name}_trace.json"

            set_run(cfg, "sample_stride", 1)
            set_run(cfg, "trajectories", ntraj)
            set_run(cfg, "base_seed", seed)
            disable_trace(cfg)
            set_output_paths(cfg, result_path, trace_path)

            write_json(cfg_path, cfg)

            manifest["cases"].append({
                "kind": "mcwf",
                "ntraj": ntraj,
                "seed": seed,
                "cfg": str(cfg_path),
                "result": str(result_path),
            })

            run_case(cfg_path, result_path, "out/1q_t1_result.json")

    write_json(OUT_DIR / "manifest.json", manifest)

    print("[package] convergence_results.tar.gz")
    ret = subprocess.run([
        "tar", "-czf", "convergence_results.tar.gz",
        "configs/convergence",
        "out/convergence",
    ])

    if ret.returncode != 0:
        raise RuntimeError("Packaging failed")

    print("Done.")
    print("Upload: convergence_results.tar.gz")


if __name__ == "__main__":
    main()