#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams


def set_paper_style():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 2.2,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def style_axis(ax, t):
    ax.set_xlim(min(t), max(t))
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.tick_params(direction="in", top=True, right=True, length=4)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)


def load_json(path):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def get_time_axis(data):
    for key in ["tlist", "times", "time"]:
        if key in data:
            return data[key]
    raise KeyError(f"No time axis found. Top-level keys: {list(data.keys())}")


def as_observable_dict(data, key_candidates):
    """
    Return observable dictionary {name: series} from several possible result
    writer conventions.
    """
    for key in key_candidates:
        if key in data:
            value = data[key]
            if isinstance(value, dict):
                return value

            names = data.get("observable_names", [])
            if isinstance(value, list) and names:
                # Support [observable][time]
                if len(value) == len(names):
                    return {name: value[i] for i, name in enumerate(names)}

                # Support [time][observable]
                if len(value) > 0 and isinstance(value[0], list):
                    return {
                        name: [row[i] for row in value]
                        for i, name in enumerate(names)
                    }

    return None


def get_ntraj(data, cli_ntraj):
    if cli_ntraj is not None:
        return cli_ntraj

    for key in ["trajectories", "ntraj", "num_trajectories", "Ntraj"]:
        if key in data:
            return int(data[key])

    run = data.get("run", {})
    if isinstance(run, dict):
        for key in ["trajectories", "ntraj", "num_trajectories", "Ntraj"]:
            if key in run:
                return int(run[key])

    return None


def get_stderr(data, ntraj=None):
    stderr = as_observable_dict(
        data,
        [
            "expectations_stderr",
            "expect_stderr",
            "stderr",
            "expectation_stderr",
            "expectations_std_error",
        ],
    )
    if stderr is not None:
        return stderr

    var = as_observable_dict(
        data,
        [
            "expectations_var",
            "expect_var",
            "variance",
            "expectations_variance",
        ],
    )
    if var is None:
        raise KeyError(
            "Could not find stderr or variance in the result file. "
            "Expected keys such as expectations_stderr, expect_stderr, "
            "expectations_var, or expect_var."
        )

    if ntraj is None:
        raise ValueError(
            "Found variance but not stderr, and the number of trajectories "
            "was not found in the result file. Re-run with stderr output or "
            "pass --ntraj."
        )

    stderr = {}
    for name, series in var.items():
        stderr[name] = [math.sqrt(max(v, 0.0) / ntraj) for v in series]
    return stderr


def pick_observables(stderr, requested):
    selected = []
    missing = []

    for name in requested:
        if name in stderr:
            selected.append(name)
        else:
            missing.append(name)

    if selected:
        if missing:
            print(f"[warn] missing requested observables: {', '.join(missing)}")
        return selected

    # Fallback: use a compact default set.
    preferred = ["P:0000", "P:1000", "Z0", "Z3", "Z1", "Z2"]
    for name in preferred:
        if name in stderr and name not in selected:
            selected.append(name)
        if len(selected) >= 4:
            break

    if selected:
        return selected

    # Last fallback: first four observables.
    return list(stderr.keys())[:4]


def max_stderr_series(stderr):
    names = list(stderr.keys())
    if not names:
        raise ValueError("No stderr series available.")

    n = len(stderr[names[0]])
    max_series = []
    max_name_series = []

    for i in range(n):
        best_name = names[0]
        best_value = stderr[best_name][i]
        for name in names[1:]:
            value = stderr[name][i]
            if value > best_value:
                best_name = name
                best_value = value
        max_series.append(best_value)
        max_name_series.append(best_name)

    return max_series, max_name_series


def pretty_label(name):
    if name.startswith("P:"):
        return rf"$P{{:}}{name[2:]}$"
    if name.startswith("Z") and len(name) > 1:
        return rf"$\langle {name}\rangle$"
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Plot MCWF standard error for a multi-trajectory FluxPulse result."
    )
    parser.add_argument("--input", default="out/4q_transport_result_multi.json")
    parser.add_argument(
        "--observables",
        nargs="*",
        default=["P:0000", "P:1000", "Z0", "Z3"],
        help="Observable names to show in the left panel.",
    )
    parser.add_argument(
        "--ntraj",
        type=int,
        default=None,
        help="Number of trajectories, only needed if the result has variance but no stderr.",
    )
    parser.add_argument("--out", default="out/fig_mcwf_uncertainty.png")
    parser.add_argument("--pdf", default="out/fig_mcwf_uncertainty.pdf")
    args = parser.parse_args()

    set_paper_style()

    data = load_json(args.input)
    t = get_time_axis(data)
    ntraj = get_ntraj(data, args.ntraj)
    stderr = get_stderr(data, ntraj=ntraj)

    selected = pick_observables(stderr, args.observables)
    max_series, max_names = max_stderr_series(stderr)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8.4, 3.25),
        constrained_layout=True
    )

    # Panel (a): selected observable stderr curves.
    for name in selected:
        axes[0].plot(t, stderr[name], label=pretty_label(name))

    axes[0].set_title("(a) Selected observables")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Standard error")
    axes[0].legend(frameon=False, loc="best")
    style_axis(axes[0], t)

    # Panel (b): max over all observables.
    axes[1].plot(t, max_series)
    axes[1].set_title("(b) Maximum over observables")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Max. standard error")
    style_axis(axes[1], t)

    # Keep y-axis starting at zero for uncertainty.
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0.0, ymax)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    if args.pdf:
        pdf_path = Path(args.pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")

    print(f"[info] plotted observables: {', '.join(selected)}")
    if ntraj is not None:
        print(f"[info] trajectories: {ntraj}")


if __name__ == "__main__":
    main()