#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def extract_series(result: dict) -> Dict[str, np.ndarray]:
    """
    Returns {name: np.array(values)} from either:
      - result["expectations"] as dict
      - result["data"] as list of {"name":..., "values":[...]}
    If both exist, "expectations" wins.
    """
    if "expectations" in result and isinstance(result["expectations"], dict):
        out = {}
        for k, v in result["expectations"].items():
            out[str(k)] = np.asarray(v, dtype=float)
        return out

    if "data" in result and isinstance(result["data"], list):
        out = {}
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            vals = item.get("values", None)
            if name and isinstance(vals, list):
                out[name] = np.asarray(vals, dtype=float)
        if out:
            return out

    raise ValueError("Could not find expectations in result. Expected dict 'expectations' or list 'data'.")


def summarize_vec(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"first": math.nan, "last": math.nan, "min": math.nan, "max": math.nan,
                "mean": math.nan, "std": math.nan}
    return {
        "first": float(x[0]),
        "last": float(x[-1]),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def minmax_time(x: np.ndarray, t: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Returns time/value at min and max. If t is None or wrong length,
    returns indices instead of times.
    """
    if x.size == 0:
        return {"t_min": math.nan, "t_max": math.nan, "min": math.nan, "max": math.nan}

    i_min = int(np.argmin(x))
    i_max = int(np.argmax(x))

    if t is not None and t.size == x.size:
        return {
            "t_min": float(t[i_min]),
            "t_max": float(t[i_max]),
            "min": float(x[i_min]),
            "max": float(x[i_max]),
        }

    # fallback: report indices
    return {
        "t_min": float(i_min),
        "t_max": float(i_max),
        "min": float(x[i_min]),
        "max": float(x[i_max]),
    }


def bounds_check(name: str, x: np.ndarray, eps: float) -> Optional[str]:
    if name.startswith("P:") or name.startswith("P0") or name.startswith("P1"):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mn < -eps or mx > 1.0 + eps:
            return f"OUT_OF_BOUNDS projector: min={mn:.6g}, max={mx:.6g} (expected [0,1])"
        return None

    head = name[:1].upper()
    if head in ("X", "Y", "Z"):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mn < -1.0 - eps or mx > 1.0 + eps:
            return f"OUT_OF_BOUNDS Pauli-like: min={mn:.6g}, max={mx:.6g} (expected [-1,1])"
        return None

    return None


def pick_names(all_names: List[str], patterns: List[str]) -> List[str]:
    if not patterns:
        return all_names

    sel = []
    for pat in patterns:
        if "*" not in pat:
            if pat in all_names:
                sel.append(pat)
            continue
        token = pat.replace("*", "")
        for n in all_names:
            if token in n and n not in sel:
                sel.append(n)
    return sel


def compute_partial_population_sum(series: Dict[str, np.ndarray], names: List[str]) -> Optional[np.ndarray]:
    if not names:
        return None
    arrs = []
    for n in names:
        if n not in series:
            continue
        arrs.append(series[n])
    if not arrs:
        return None
    L = min(a.size for a in arrs)
    arrs = [a[:L] for a in arrs]
    return np.sum(np.vstack(arrs), axis=0)


def main():
    ap = argparse.ArgumentParser(description="Compute stats for a fluxpulse result JSON (large-file friendly).")
    ap.add_argument("--result", required=True, help="Path to result JSON")
    ap.add_argument("--only", default="", help="Comma-separated list of names or wildcard patterns (e.g. Z0,Z1,P:00000,P:)")
    ap.add_argument("--max", type=int, default=50, help="Max number of observables to print")
    ap.add_argument("--eps", type=float, default=1e-6, help="Tolerance for bounds checks")
    ap.add_argument("--pop_sum", default="",
                    help="Comma-separated projector names to sum (e.g. P:00000,P:10000,P:01000,P:00100,P:00010,P:00001)")
    args = ap.parse_args()

    result = load_json(args.result)

    t = np.asarray(result.get("tlist", []), dtype=float)
    if t.size == 0:
        print("[stats] WARNING: no tlist found (or empty). Will still summarize series (times=indices).")
        t_use = None
    else:
        print(f"[stats] times: n={t.size}  t0={t[0]:.6g}  t_end={t[-1]:.6g}")
        t_use = t

    series = extract_series(result)
    names_all = sorted(series.keys())
    print(f"[stats] observables found: {len(names_all)}")

    patterns = [p.strip() for p in args.only.split(",") if p.strip()]
    names = pick_names(names_all, patterns)

    if not names:
        print("[stats] No observables matched --only filter.")
        return

    names = names[: max(1, int(args.max))]

    any_warn = False
    for n in names:
        x = series[n]
        s = summarize_vec(x)
        mm = minmax_time(x, t_use)
        warn = bounds_check(n, x, args.eps)

        line = (f"[{n}] first={s['first']:.6g} last={s['last']:.6g} "
                f"mean={s['mean']:.6g} std={s['std']:.6g}  "
                f"min={mm['min']:.6g}@t={mm['t_min']:.6g}  "
                f"max={mm['max']:.6g}@t={mm['t_max']:.6g}")
        if warn:
            any_warn = True
            line += f"  <-- {warn}"
        print(line)

    pop_list = [p.strip() for p in args.pop_sum.split(",") if p.strip()]
    if pop_list:
        y = compute_partial_population_sum(series, pop_list)
        if y is None:
            print("[stats] pop_sum: none of the requested series were found.")
        else:
            s = summarize_vec(y)
            mm = minmax_time(y, t_use)
            print("[stats] pop_sum:")
            print(f"  names={pop_list}")
            print(f"  first={s['first']:.6g} last={s['last']:.6g} mean={s['mean']:.6g} std={s['std']:.6g}")
            print(f"  min={mm['min']:.6g}@t={mm['t_min']:.6g}  max={mm['max']:.6g}@t={mm['t_max']:.6g}")

    if any_warn:
        print("[stats] NOTE: bounds warnings may occur for strong drives + single trajectory, "
              "but large violations usually indicate a bug.")
    else:
        print("[stats] OK: no bounds warnings for printed series.")


if __name__ == "__main__":
    main()