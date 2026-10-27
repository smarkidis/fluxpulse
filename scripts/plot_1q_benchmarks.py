#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams


def load_result(path):
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

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
    else:
        names = data.get("observable_names", [])
        values = data.get("expectations", data.get("expect", None))

        if values is None or not names:
            raise KeyError(f"No observable data found in {path}")

        # Support both [observable][time] and [time][observable].
        if len(values) == len(names):
            obs = {name: values[i] for i, name in enumerate(names)}
        else:
            obs = {
                name: [row[i] for row in values]
                for i, name in enumerate(names)
            }

    return t, obs


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


def main():
    parser = argparse.ArgumentParser(
        description="Plot paper-style one-qubit benchmark figure."
    )
    parser.add_argument("--rabi", default="out/1q_rabi_result.json")
    parser.add_argument("--ramsey", default="out/1q_ramsey_result.json")
    parser.add_argument("--t1", default="out/1q_t1_result.json")
    parser.add_argument("--out", default="out/fig_1q_benchmarks.png")
    parser.add_argument("--pdf", default="out/fig_1q_benchmarks.pdf")
    args = parser.parse_args()

    set_paper_style()

    t_rabi, obs_rabi = load_result(args.rabi)
    t_ramsey, obs_ramsey = load_result(args.ramsey)
    t_t1, obs_t1 = load_result(args.t1)

    _, rabi_p1 = pick_observable(
        obs_rabi,
        ["P1", "P:1", "P_1"],
        "Rabi excited-state population",
    )

    _, ramsey_x = pick_observable(
        obs_ramsey,
        ["X", "X0"],
        "Ramsey X expectation",
    )

    _, ramsey_y = pick_observable(
        obs_ramsey,
        ["Y", "Y0"],
        "Ramsey Y expectation",
    )

    t1_name, t1_signal = pick_observable(
        obs_t1,
        ["P1", "P:1", "P_1", "Z", "Z0"],
        "T1 signal",
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(11.5, 3.25),
        constrained_layout=True
    )

    # Rabi
    axes[0].plot(t_rabi, rabi_p1)
    axes[0].set_title("(a) Rabi")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$P_1(t)$")
    axes[0].set_ylim(-0.05, 1.05)
    style_axis(axes[0], t_rabi)

    # Ramsey
    axes[1].plot(t_ramsey, ramsey_x, label=r"$\langle X\rangle$")
    axes[1].plot(t_ramsey, ramsey_y, label=r"$\langle Y\rangle$")
    axes[1].set_title("(b) Ramsey")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Expectation")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(frameon=False, loc="best")
    style_axis(axes[1], t_ramsey)

    # T1
    axes[2].plot(t_t1, t1_signal)
    axes[2].set_title(r"(c) $T_1$ relaxation")
    axes[2].set_xlabel("Time")

    if t1_name in ["Z", "Z0"]:
        axes[2].set_ylabel(r"$\langle Z(t)\rangle$")
        axes[2].set_ylim(-1.05, 1.05)
    else:
        axes[2].set_ylabel(r"$P_1(t)$")
        axes[2].set_ylim(-0.05, 1.05)

    style_axis(axes[2], t_t1)

    # In a paper, the caption usually replaces a large global title.
    # Uncomment if you want a title:
    # fig.suptitle("Single-qubit pulse benchmarks", fontsize=13, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    if args.pdf:
        pdf_path = Path(args.pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()