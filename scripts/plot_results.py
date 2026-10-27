#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str):
    return json.loads(Path(path).read_text())


def _normalize_names(result: dict, nobs: int) -> list[str]:
    # Prefer "observable_names", then "observables", then dict keys, else fallback.
    for k in ["observable_names", "observables", "names"]:
        v = result.get(k)
        if isinstance(v, list) and v:
            names = [str(x) for x in v]
            if len(names) >= nobs:
                return names[:nobs]
            return names + [f"obs{i}" for i in range(len(names), nobs)]
    return [f"obs{i}" for i in range(nobs)]


def get_expectations(result: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Returns:
      t: (nt,)
      names: (nobs,)
      mat: (nobs, nt)
    Supports:
      - expectations as dict {name: series}
      - expectations as list-of-lists [obs][time]
      - expectations as flat list length nobs*nt
    """
    t = np.asarray(result.get("tlist", []), dtype=float)
    if t.size == 0:
        raise ValueError("Result JSON missing or empty 'tlist'.")

    E = result.get("expectations", None)
    if E is None:
        raise ValueError("Result JSON missing 'expectations'.")

    nt = t.size

    # Case 1: dict of series
    if isinstance(E, dict):
        # Keep insertion order of dict (Python 3.7+ preserves).
        names = [str(k) for k in E.keys()]
        series = []
        for k in names:
            y = np.asarray(E[k], dtype=float)
            if y.size != nt:
                raise ValueError(f"expectations['{k}'] length {y.size} != nt={nt}")
            series.append(y)
        mat = np.vstack(series)
        return t, names, mat

    # Case 2: list-of-lists matrix
    if isinstance(E, list) and len(E) > 0 and isinstance(E[0], list):
        mat = np.asarray(E, dtype=float)
        if mat.ndim != 2 or mat.shape[1] != nt:
            raise ValueError(f"Expectations matrix has shape {mat.shape}, expected (*, {nt})")
        names = _normalize_names(result, mat.shape[0])
        return t, names, mat

    # Case 3: flat list
    flat = np.asarray(E, dtype=float).ravel()
    if flat.size % nt != 0:
        raise ValueError(f"Flattened expectations length {flat.size} not divisible by nt={nt}")
    nobs = flat.size // nt
    mat = flat.reshape((nobs, nt))
    names = _normalize_names(result, nobs)
    return t, names, mat


def try_infer_swap_overlay(config: dict, t: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    model = config.get("model", {})
    mtype = str(model.get("type", "")).lower()
    if mtype not in ["coupled_qubits", "coupled", "2q", "multi_qubit"]:
        return {}

    exch = model.get("exchange_xy", [])
    zz = model.get("zz", [])
    if not exch or zz:
        return {}
    if not (isinstance(exch, list) and len(exch) == 1):
        return {}
    c = exch[0]
    if not (c.get("q0") == 0 and c.get("q1") == 1):
        return {}
    J = float(c.get("strength", 0.0))
    if J == 0.0:
        return {}

    out = {}
    if "P:01" in names:
        out["P:01 (theory)"] = np.sin(2.0 * J * t) ** 2
    if "P:10" in names:
        out["P:10 (theory)"] = np.cos(2.0 * J * t) ** 2
    return out


def try_infer_zz_overlay(config: dict, t: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    model = config.get("model", {})
    mtype = str(model.get("type", "")).lower()
    if mtype not in ["coupled_qubits", "coupled", "2q", "multi_qubit"]:
        return {}

    zz = model.get("zz", [])
    exch = model.get("exchange_xy", [])
    if not zz or exch:
        return {}
    if not (isinstance(zz, list) and len(zz) == 1):
        return {}
    c = zz[0]
    if not (c.get("q0") == 0 and c.get("q1") == 1):
        return {}
    zeta = float(c.get("strength", 0.0))
    if zeta == 0.0:
        return {}

    out = {}
    if "X0" in names:
        out["X0 (theory)"] = np.cos(2.0 * zeta * t)
    if "X1" in names:
        out["X1 (theory)"] = np.cos(2.0 * zeta * t)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot fluxpulse result JSON (optionally using config JSON for overlays).")
    ap.add_argument("--config", type=str, default=None, help="Path to config JSON (optional).")
    ap.add_argument("--result", type=str, required=True, help="Path to result JSON.")
    ap.add_argument("--out", type=str, default=None, help="If set, save plot to this path (png/pdf).")
    ap.add_argument("--show", action="store_true", help="Show interactive plot window.")
    ap.add_argument("--max_obs", type=int, default=16, help="Max observables to plot (avoid clutter).")
    ap.add_argument("--downsample", type=int, default=1, help="Plot every k-th point for speed (>=1).")
    args = ap.parse_args()

    result = load_json(args.result)
    config = load_json(args.config) if args.config else None

    t, names, mat = get_expectations(result)

    # Downsample for plotting
    k = max(1, int(args.downsample))
    t_plot = t[::k]
    mat_plot = mat[:, ::k]

    nplot = min(mat_plot.shape[0], max(1, int(args.max_obs)))

    plt.figure()
    for i in range(nplot):
        plt.plot(t_plot, mat_plot[i], label=names[i])

    # Optional overlays
    if config is not None:
        overlays = {}
        overlays.update(try_infer_swap_overlay(config, t_plot, names))
        overlays.update(try_infer_zz_overlay(config, t_plot, names))
        for label, y in overlays.items():
            plt.plot(t_plot, y, linestyle="--", label=label)

    plt.xlabel("time")
    plt.ylabel("expectation")
    plt.title(Path(args.result).name)
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to: {out_path}")

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()