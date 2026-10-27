#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _normalize_names(result: dict, nobs: int) -> List[str]:
    # Prefer "observable_names", then "observables", then "names", else fallback.
    for k in ["observable_names", "observables", "names"]:
        v = result.get(k)
        if isinstance(v, list) and v:
            names = [str(x) for x in v]
            if len(names) >= nobs:
                return names[:nobs]
            return names + [f"obs{i}" for i in range(len(names), nobs)]
    return [f"obs{i}" for i in range(nobs)]


def get_series_matrix(field: str, t: np.ndarray, result: dict) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """Read a field that may be dict {name: series}, matrix [obs][time], or flat list."""
    nt = t.size
    obj = result.get(field, None)
    if obj is None:
        return None, None

    # Case 1: dict of series
    if isinstance(obj, dict):
        names = [str(k) for k in obj.keys()]
        series = []
        for k in names:
            y = np.asarray(obj[k], dtype=float)
            if y.size != nt:
                raise ValueError(f"{field}['{k}'] length {y.size} != nt={nt}")
            series.append(y)
        return names, np.vstack(series)

    # Case 2: list-of-lists matrix
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
        mat = np.asarray(obj, dtype=float)
        if mat.ndim != 2 or mat.shape[1] != nt:
            raise ValueError(f"{field} matrix has shape {mat.shape}, expected (*, {nt})")
        names = _normalize_names(result, mat.shape[0])
        return names, mat

    # Case 3: flat list
    flat = np.asarray(obj, dtype=float).ravel()
    if flat.size % nt != 0:
        raise ValueError(f"Flattened {field} length {flat.size} not divisible by nt={nt}")
    nobs = flat.size // nt
    mat = flat.reshape((nobs, nt))
    names = _normalize_names(result, nobs)
    return names, mat


def get_expectations(result: dict) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return t (nt,), names (nobs,), mean matrix (nobs, nt)."""
    t = np.asarray(result.get("tlist", []), dtype=float)
    if t.size == 0:
        raise ValueError("Result JSON missing or empty 'tlist'.")

    E = result.get("expectations", None)
    if E is None:
        raise ValueError("Result JSON missing 'expectations'.")

    nt = t.size

    # dict
    if isinstance(E, dict):
        names = [str(k) for k in E.keys()]
        series = []
        for k in names:
            y = np.asarray(E[k], dtype=float)
            if y.size != nt:
                raise ValueError(f"expectations['{k}'] length {y.size} != nt={nt}")
            series.append(y)
        mat = np.vstack(series)
        return t, names, mat

    # matrix
    if isinstance(E, list) and len(E) > 0 and isinstance(E[0], list):
        mat = np.asarray(E, dtype=float)
        if mat.ndim != 2 or mat.shape[1] != nt:
            raise ValueError(f"Expectations matrix has shape {mat.shape}, expected (*, {nt})")
        names = _normalize_names(result, mat.shape[0])
        return t, names, mat

    # flat
    flat = np.asarray(E, dtype=float).ravel()
    if flat.size % nt != 0:
        raise ValueError(f"Flattened expectations length {flat.size} not divisible by nt={nt}")
    nobs = flat.size // nt
    mat = flat.reshape((nobs, nt))
    names = _normalize_names(result, nobs)
    return t, names, mat


def get_uncertainty(result: dict, t: np.ndarray, mean_names: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (var_mat, stderr_mat), aligned to mean_names order when possible."""

    var_names, var_mat = None, None
    se_names, se_mat = None, None

    for key in ["expect_var", "expectations_var"]:
        var_names, var_mat = get_series_matrix(key, t, result)
        if var_mat is not None:
            break

    for key in ["expect_stderr", "expectations_stderr"]:
        se_names, se_mat = get_series_matrix(key, t, result)
        if se_mat is not None:
            break

    # Fallback: per-entry data list.
    if (var_mat is None or se_mat is None) and isinstance(result.get("data"), list):
        data_list = result["data"]

        if var_mat is None:
            var_map: Dict[str, np.ndarray] = {}
            for item in data_list:
                if isinstance(item, dict) and "name" in item and "var" in item:
                    y = np.asarray(item["var"], dtype=float)
                    if y.size == t.size:
                        var_map[str(item["name"])] = y
            if var_map:
                var_names = list(var_map.keys())
                var_mat = np.vstack([var_map[n] for n in var_names])

        if se_mat is None:
            se_map: Dict[str, np.ndarray] = {}
            for item in data_list:
                if isinstance(item, dict) and "name" in item and "stderr" in item:
                    y = np.asarray(item["stderr"], dtype=float)
                    if y.size == t.size:
                        se_map[str(item["name"])] = y
            if se_map:
                se_names = list(se_map.keys())
                se_mat = np.vstack([se_map[n] for n in se_names])

    def _align(names: Optional[List[str]], mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mat is None or names is None:
            return None
        if names == mean_names:
            return mat
        idx = {n: i for i, n in enumerate(names)}
        rows = []
        for n in mean_names:
            if n in idx:
                rows.append(mat[idx[n]])
            else:
                rows.append(np.full((t.size,), np.nan, dtype=float))
        return np.vstack(rows)

    return _align(var_names, var_mat), _align(se_names, se_mat)


def analyze(t: np.ndarray, names: List[str], mean: np.ndarray,
            var: Optional[np.ndarray], stderr: Optional[np.ndarray]) -> None:
    print("=== Uncertainty analysis ===")
    print(f"observables={len(names)}  times={t.size}")

    if var is None and stderr is None:
        print("No variance/stderr found in result JSON.")
        print("Expected keys: expect_var/expect_stderr (or expectations_var/expectations_stderr) or data[].var/stderr")
        return

    # Relative stderr can blow up when the mean crosses 0. Use a small floor so
    # the "worst_rel" metric remains informative.
    rel_floor = 5e-2
    eps = 1e-12
    worst_rel_se = (None, -np.inf, None)  # name, value, time
    worst_var = (None, -np.inf, None)

    for i, name in enumerate(names):
        mean_i = mean[i]
        var_i = var[i] if var is not None else None
        se_i = stderr[i] if stderr is not None else None

        if var_i is not None and not np.all(np.isnan(var_i)):
            vmax_idx = int(np.nanargmax(var_i))
            vmax = float(var_i[vmax_idx])
            vmean = float(np.nanmean(var_i))
            vfinal = float(var_i[-1])
            # numpy.trapz is deprecated in newer NumPy; trapezoid is the replacement.
            vint = float(np.trapezoid(np.nan_to_num(var_i, nan=0.0), t))
            print(f"- {name}: var mean={vmean:.6g}  max={vmax:.6g} @t={t[vmax_idx]:.6g}  final={vfinal:.6g}  int={vint:.6g}")
            if vmax > worst_var[1]:
                worst_var = (name, vmax, float(t[vmax_idx]))
        else:
            print(f"- {name}: var unavailable")

        if se_i is not None and not np.all(np.isnan(se_i)):
            denom = np.maximum(np.abs(mean_i), rel_floor)
            rel = np.abs(se_i) / (denom + eps)
            ridx = int(np.nanargmax(rel))
            rmax = float(rel[ridx])
            semax = float(se_i[int(np.nanargmax(se_i))])
            semean = float(np.nanmean(se_i))
            mean_at = float(mean_i[ridx])
            print(f"  stderr mean={semean:.6g}  max={semax:.6g}  worst_rel={rmax:.6g} @t={t[ridx]:.6g} (mean={mean_at:.6g}, floor={rel_floor:g})")
            if rmax > worst_rel_se[1]:
                worst_rel_se = (name, rmax, float(t[ridx]))
        elif stderr is not None:
            print("  stderr unavailable")

    if worst_var[0] is not None:
        print(f"Worst variance: {worst_var[0]}  max_var={worst_var[1]:.6g} @t={worst_var[2]:.6g}")
    if worst_rel_se[0] is not None:
        print(f"Worst relative stderr: {worst_rel_se[0]}  max_rel={worst_rel_se[1]:.6g} @t={worst_rel_se[2]:.6g}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot and analyze fluxpulse variance/stderr from result JSON.")
    ap.add_argument("--result", type=str, required=True, help="Path to result JSON.")
    ap.add_argument("--out", type=str, default=None, help="If set, save plot to this path (png/pdf).")
    ap.add_argument("--show", action="store_true", help="Show interactive plot window.")
    ap.add_argument("--mode", choices=["var", "stderr", "both", "band"], default="var",
                    help="What to plot: var, stderr, both (two figures), band (mean with +/-2*stderr band).")
    ap.add_argument("--max_obs", type=int, default=16, help="Max observables to plot (avoid clutter).")
    ap.add_argument("--downsample", type=int, default=1, help="Plot every k-th point for speed (>=1).")
    ap.add_argument("--no_analysis", action="store_true", help="Disable printing uncertainty analysis to stdout.")
    args = ap.parse_args()

    result = load_json(args.result)
    t, names, mean = get_expectations(result)
    var, stderr = get_uncertainty(result, t, names)

    if not args.no_analysis:
        analyze(t, names, mean, var, stderr)

    k = max(1, int(args.downsample))
    t_plot = t[::k]
    mean_plot = mean[:, ::k]
    var_plot = var[:, ::k] if var is not None else None
    se_plot = stderr[:, ::k] if stderr is not None else None

    nplot = min(len(names), max(1, int(args.max_obs)))

    def save(fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to: {path}")

    if args.mode in ["var", "stderr"]:
        plt.figure()
        if args.mode == "var":
            if var_plot is None:
                raise SystemExit("No variance found in result JSON.")
            for i in range(nplot):
                plt.plot(t_plot, var_plot[i], label=names[i])
            plt.ylabel("variance")
            plt.title(Path(args.result).name + " (variance)")
        else:
            if se_plot is None:
                raise SystemExit("No stderr found in result JSON.")
            for i in range(nplot):
                plt.plot(t_plot, se_plot[i], label=names[i])
            plt.ylabel("stderr")
            plt.title(Path(args.result).name + " (stderr)")

        plt.xlabel("time")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)

        if args.out:
            save(plt.gcf(), Path(args.out))
        if args.show or not args.out:
            plt.show()

    elif args.mode == "both":
        base_out = Path(args.out) if args.out else None

        if var_plot is None:
            raise SystemExit("No variance found in result JSON.")
        plt.figure()
        for i in range(nplot):
            plt.plot(t_plot, var_plot[i], label=names[i])
        plt.xlabel("time")
        plt.ylabel("variance")
        plt.title(Path(args.result).name + " (variance)")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)
        if base_out is not None:
            save(plt.gcf(), base_out.with_name(base_out.stem + "_var" + base_out.suffix))
        if args.show and base_out is None:
            plt.show()

        if se_plot is None:
            raise SystemExit("No stderr found in result JSON.")
        plt.figure()
        for i in range(nplot):
            plt.plot(t_plot, se_plot[i], label=names[i])
        plt.xlabel("time")
        plt.ylabel("stderr")
        plt.title(Path(args.result).name + " (stderr)")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)
        if base_out is not None:
            save(plt.gcf(), base_out.with_name(base_out.stem + "_stderr" + base_out.suffix))
        if args.show or base_out is None:
            plt.show()

    elif args.mode == "band":
        if se_plot is None:
            raise SystemExit("Mode 'band' requires stderr in result JSON.")
        plt.figure()
        for i in range(nplot):
            m = mean_plot[i]
            s = se_plot[i]
            plt.plot(t_plot, m, label=names[i])
            plt.fill_between(t_plot, m - 2.0 * s, m + 2.0 * s, alpha=0.2)
        plt.xlabel("time")
        plt.ylabel("expectation")
        plt.title(Path(args.result).name + " (mean ± 2·stderr)")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)
        if args.out:
            save(plt.gcf(), Path(args.out))
        if args.show or not args.out:
            plt.show()


if __name__ == "__main__":
    main()
