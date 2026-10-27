#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams


def resolve_result_path(path):
    """If a trace path is passed, try the matching result path."""
    path = Path(path)
    if path.exists() and "traceEvents" not in path.read_text(errors="ignore")[:500]:
        return path

    if "trace" in path.name:
        candidate = path.with_name(path.name.replace("_trace", "_result"))
        if candidate.exists():
            print(f"[info] using result file instead of trace: {candidate}")
            return candidate

    return path


def load_result(path):
    path = resolve_result_path(path)

    with path.open("r") as f:
        data = json.load(f)

    if "traceEvents" in data:
        raise ValueError(
            f"{path} appears to be a trace file, not a result file. "
            "Use the matching *_result.json file for observable plots."
        )

    if "tlist" in data:
        t = data["tlist"]
    elif "times" in data:
        t = data["times"]
    elif "time" in data:
        t = data["time"]
    else:
        raise KeyError(f"No time axis found in {path}")

    if "expectations" in data and isinstance(data["expectations"], dict):
        obs = data["expectations"]
    elif "expect" in data and isinstance(data["expect"], dict):
        obs = data["expect"]
    elif "expectations_mean" in data and isinstance(data["expectations_mean"], dict):
        obs = data["expectations_mean"]
    else:
        names = data.get("observable_names", [])
        values = data.get("expectations", data.get("expect", None))

        if values is None or not names:
            raise KeyError(
                f"No observable data found in {path}. "
                f"Top-level keys are: {list(data.keys())}"
            )

        # Support both [observable][time] and [time][observable].
        if len(values) == len(names):
            obs = {name: values[i] for i, name in enumerate(names)}
        else:
            obs = {name: [row[i] for row in values] for i, name in enumerate(names)}

    return t, obs, path


def pick_observable(obs, candidates, label):
    for name in candidates:
        if name in obs:
            return name, obs[name]

    available = ", ".join(obs.keys())
    raise KeyError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available observables: {available}"
    )


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
        "lines.linewidth": 2.3,
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot paper-style two-qubit SWAP/exchange and ZZ benchmarks."
    )
    parser.add_argument(
        "--swap",
        default="out/2q_swap_result.json",
        help="Path to 2q_swap_result.json. If a trace file is given, the script tries the matching result file.",
    )
    parser.add_argument(
        "--zz",
        default="out/2q_zz_result.json",
        help="Path to 2q_zz_result.json.",
    )
    parser.add_argument("--out", default="out/fig_2q_benchmarks.png")
    parser.add_argument("--pdf", default="out/fig_2q_benchmarks.pdf")
    args = parser.parse_args()

    set_paper_style()

    t_swap, obs_swap, swap_path = load_result(args.swap)
    t_zz, obs_zz, zz_path = load_result(args.zz)

    # SWAP / exchange: population transfer between |10> and |01>.
    p10_name, p10 = pick_observable(
        obs_swap,
        ["P:10", "P10", "P:010", "P:0010"],
        "SWAP population P:10",
    )
    p01_name, p01 = pick_observable(
        obs_swap,
        ["P:01", "P01", "P:001", "P:0001"],
        "SWAP population P:01",
    )

    # ZZ phase: X0 and X1 are phase-sensitive observables.
    x0_name, x0 = pick_observable(obs_zz, ["X0", "X"], "ZZ observable X0")
    x1_name, x1 = pick_observable(obs_zz, ["X1"], "ZZ observable X1")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8.2, 3.25),
        constrained_layout=True
    )

    # Panel (a): SWAP / exchange
    axes[0].plot(t_swap, p10, label=r"$P{:}10$")
    axes[0].plot(t_swap, p01, label=r"$P{:}01$", linestyle="--")
    axes[0].set_title("(a) Exchange")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False, loc="best")
    style_axis(axes[0], t_swap)

    # Panel (b): ZZ phase accumulation
    axes[1].plot(t_zz, x0, label=r"$\langle X_0\rangle$")
    axes[1].plot(t_zz, x1, label=r"$\langle X_1\rangle$", linestyle="--")
    axes[1].set_title(r"(b) $ZZ$ phase")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Expectation")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(frameon=False, loc="best")
    style_axis(axes[1], t_zz)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    if args.pdf:
        pdf_path = Path(args.pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")

    print(f"[info] SWAP source: {swap_path}")
    print(f"[info] ZZ source:   {zz_path}")


if __name__ == "__main__":
    main()