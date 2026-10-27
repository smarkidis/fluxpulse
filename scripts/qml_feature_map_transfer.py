#!/usr/bin/env python3
"""Evaluate transfer of a frozen pulse-level quantum feature map.

The source pulse map is kept fixed.  On each target split, only the QSVM
coefficients are fitted.  This tests transfer of the learned representation,
not zero-shot transfer of a classifier.  A target-optimized pulse map is
evaluated on the same splits as a reference.

The script reconstructs the exact two-qubit, one-window evolution from the
dimensionless parameters stored in best_encoding.json and produces:

  * a two-panel target decision-boundary figure (PDF, PNG, and SVG),
  * a JSON summary over all target seeds,
  * an NPZ file containing the displayed boundary data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_VERSION = "2026-08-15-feature-map-transfer-v2"

BLUE = "#0065BD"
RED = "#D62728"
REGION_BLUE = "#DCEBFA"
REGION_RED = "#F8DEDE"


@dataclass(frozen=True)
class DatasetSplit:
    X_dev: np.ndarray
    X_test: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class MapEvaluation:
    test_accuracy: float
    test_prediction: np.ndarray
    K_dev: np.ndarray
    K_test: np.ndarray
    decision: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer one frozen entangling pulse map to target datasets and "
            "compare it with target-optimized maps."
        )
    )
    parser.add_argument(
        "--source-best",
        required=True,
        type=Path,
        help="Source-domain entangling best_encoding.json.",
    )
    parser.add_argument(
        "--target-root",
        required=True,
        type=Path,
        help="Target experiment root containing seed_*/... results.",
    )
    parser.add_argument(
        "--target-family",
        default="2q_entangling_encoding",
        help="Target result family used as the optimized reference.",
    )
    parser.add_argument(
        "--fluxpulse",
        type=Path,
        help="Path to the FluxPulse executable.",
    )
    parser.add_argument(
        "--fluxpulse-helper",
        type=Path,
        default=None,
        help=(
            "Path to qml_time_convergence.py. By default, use the copy "
            "beside this script."
        ),
    )
    parser.add_argument("--integration-intervals", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--analytic-only",
        action="store_true",
        help=(
            "Use direct exponentiation for a quick check. Omit this option "
            "for the paper run, which evaluates transferred states with "
            "FluxPulse."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute transferred states even when a cache is present.",
    )
    parser.add_argument(
        "--display-seed",
        type=int,
        default=None,
        help="Target seed whose decision boundaries are displayed.",
    )
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=260)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out/qml/feature_map_transfer"),
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    for key in ("best_params", "dataset", "dataset_metadata", "seed"):
        if key not in summary:
            raise ValueError(f"{path}: missing required key {key!r}")
    return summary


def load_fluxpulse_helper(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"cannot find {path}; place qml_time_convergence.py beside this "
            "script or pass --fluxpulse-helper"
        )
    spec = importlib.util.spec_from_file_location("qml_transfer_fluxpulse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import FluxPulse helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "params_from_summary",
        "pulse_duration_from_summary",
        "resolve_template_path",
        "FluxPulseEncoder",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"{path}: missing helper functions {missing}")
    return module


def make_spirals(metadata: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    seed = int(metadata["seed"])
    samples = int(metadata["samples"])
    turns = float(metadata["spiral_turns"])
    noise = float(metadata["spiral_noise"])
    radial_floor = float(metadata.get("spiral_radial_floor", 0.03))

    rng = np.random.default_rng(seed)
    class_sizes = (samples // 2, samples - samples // 2)

    def arm(n_points: int, phase: float) -> np.ndarray:
        radius = np.sqrt(rng.uniform(radial_floor, 1.0, size=n_points))
        angle = 2.0 * math.pi * turns * radius + phase
        points = np.column_stack(
            (radius * np.cos(angle), radius * np.sin(angle))
        )
        if noise > 0.0:
            points += rng.normal(0.0, noise, size=points.shape)
        return points

    X = np.vstack((arm(class_sizes[0], 0.0), arm(class_sizes[1], math.pi)))
    y = np.concatenate(
        (
            np.zeros(class_sizes[0], dtype=np.int64),
            np.ones(class_sizes[1], dtype=np.int64),
        )
    )
    permutation = rng.permutation(samples)
    return X[permutation], y[permutation]


def reconstruct_split(summary: Mapping[str, object]) -> DatasetSplit:
    metadata = summary["dataset_metadata"]
    dataset = str(summary["dataset"])
    seed = int(summary["seed"])
    samples = int(metadata["samples"])

    if dataset == "moons":
        X, y = make_moons(
            n_samples=samples,
            noise=float(metadata["moons_noise"]),
            random_state=seed,
        )
    elif dataset == "spirals":
        X, y = make_spirals(metadata)
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")

    test_samples = int(metadata["test_samples"])
    X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_samples,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_dev_std = scaler.fit_transform(X_dev_raw)
    X_test_std = scaler.transform(X_test_raw)
    maxabs = np.max(np.abs(X_dev_std), axis=0)
    maxabs[maxabs < 1.0e-12] = 1.0

    def scale(values: np.ndarray) -> np.ndarray:
        return np.clip(values / maxabs, -1.0, 1.0)

    return DatasetSplit(
        X_dev=scale(X_dev_std),
        X_test=scale(X_test_std),
        y_dev=np.asarray(y_dev, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
    )


def pauli_operators() -> tuple[np.ndarray, ...]:
    identity = np.eye(2, dtype=np.complex128)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    return (
        np.kron(sigma_x, identity),
        np.kron(identity, sigma_x),
        np.kron(sigma_y, identity),
        np.kron(identity, sigma_y),
        np.kron(sigma_z, identity),
        np.kron(identity, sigma_z),
        np.kron(sigma_z, sigma_z),
    )


X0, X1, Y0, Y1, Z0, Z1, ZZ = pauli_operators()


def interaction_angle(best_params: Mapping[str, object]) -> float:
    dimensionless = best_params["dimensionless"]
    if dimensionless.get("chi") is not None:
        return float(dimensionless["chi"])

    physical = best_params.get("physical", {})
    tau = float(physical["pulse_duration"])
    angle = 0.0
    for edge in physical.get("reference_zz_edges", []):
        if edge.get("angle") is not None:
            angle += float(edge["angle"])
        else:
            angle += float(edge["strength"]) * tau
    return angle


def encode_states(
    points: np.ndarray,
    best_params: Mapping[str, object],
    batch_size: int = 8192,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("this transfer script requires two-dimensional inputs")

    dimensionless = best_params["dimensionless"]
    alpha = np.asarray(dimensionless["alpha"], dtype=np.float64)
    delta = np.asarray(dimensionless["delta"], dtype=np.float64)
    phase = np.asarray(
        dimensionless.get("phase", np.zeros_like(alpha)),
        dtype=np.float64,
    )
    if alpha.shape != (2,) or delta.shape != (2,) or phase.shape != (2,):
        raise ValueError("the stored map must contain two alpha/delta/phase values")

    transverse0 = math.cos(phase[0]) * X0 + math.sin(phase[0]) * Y0
    transverse1 = math.cos(phase[1]) * X1 + math.sin(phase[1]) * Y1
    chi = interaction_angle(best_params)

    states_out = np.empty((len(points), 4), dtype=np.complex128)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        block = points[start:stop]
        generator = 0.5 * (
            alpha[0] * block[:, 0, None, None] * transverse0
            + alpha[1] * block[:, 1, None, None] * transverse1
            + delta[0] * block[:, 0, None, None] * Z0
            + delta[1] * block[:, 1, None, None] * Z1
        )
        if chi != 0.0:
            generator = generator + chi * ZZ

        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        initial_coefficients = np.conjugate(eigenvectors[:, 0, :])
        evolved_coefficients = np.exp(-1j * eigenvalues) * initial_coefficients
        states = np.einsum(
            "bij,bj->bi",
            eigenvectors,
            evolved_coefficients,
            optimize=True,
        )
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        states_out[start:stop] = states
    return states_out


def fidelity_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left @ np.conjugate(right).T) ** 2


def evaluate_map(
    split: DatasetSplit,
    best_params: Mapping[str, object],
    svm_c: float,
    grid_points: np.ndarray | None = None,
) -> MapEvaluation:
    dev_states = encode_states(split.X_dev, best_params)
    test_states = encode_states(split.X_test, best_params)
    K_dev = fidelity_kernel(dev_states, dev_states)
    K_test = fidelity_kernel(test_states, dev_states)

    classifier = SVC(kernel="precomputed", C=svm_c)
    classifier.fit(K_dev, split.y_dev)
    prediction = classifier.predict(K_test)
    accuracy = float(np.mean(prediction == split.y_test))

    decision = None
    if grid_points is not None:
        decision = np.empty(len(grid_points), dtype=np.float64)
        batch_size = 4096
        for start in range(0, len(grid_points), batch_size):
            stop = min(start + batch_size, len(grid_points))
            grid_states = encode_states(grid_points[start:stop], best_params)
            K_grid = fidelity_kernel(grid_states, dev_states)
            decision[start:stop] = classifier.decision_function(K_grid)

    return MapEvaluation(
        test_accuracy=accuracy,
        test_prediction=np.asarray(prediction, dtype=np.int64),
        K_dev=K_dev,
        K_test=K_test,
        decision=decision,
    )


def evaluate_state_arrays(
    split: DatasetSplit,
    dev_states: np.ndarray,
    test_states: np.ndarray,
    best_params: Mapping[str, object],
    svm_c: float,
    grid_points: np.ndarray | None = None,
) -> MapEvaluation:
    if dev_states.shape != (len(split.y_dev), 4):
        raise ValueError("development-state array has the wrong shape")
    if test_states.shape != (len(split.y_test), 4):
        raise ValueError("test-state array has the wrong shape")

    K_dev = fidelity_kernel(dev_states, dev_states)
    K_test = fidelity_kernel(test_states, dev_states)
    classifier = SVC(kernel="precomputed", C=svm_c)
    classifier.fit(K_dev, split.y_dev)
    prediction = classifier.predict(K_test)
    accuracy = float(np.mean(prediction == split.y_test))

    decision = None
    if grid_points is not None:
        decision = np.empty(len(grid_points), dtype=np.float64)
        batch_size = 4096
        for start in range(0, len(grid_points), batch_size):
            stop = min(start + batch_size, len(grid_points))
            grid_states = encode_states(grid_points[start:stop], best_params)
            K_grid = fidelity_kernel(grid_states, dev_states)
            decision[start:stop] = classifier.decision_function(K_grid)

    return MapEvaluation(
        test_accuracy=accuracy,
        test_prediction=np.asarray(prediction, dtype=np.int64),
        K_dev=K_dev,
        K_test=K_test,
        decision=decision,
    )


def analytic_kernel_error(
    split: DatasetSplit,
    best_params: Mapping[str, object],
    reference_kernel: np.ndarray,
) -> float:
    analytic_states = encode_states(split.X_dev, best_params)
    analytic_kernel = fidelity_kernel(analytic_states, analytic_states)
    return float(
        np.linalg.norm(analytic_kernel - reference_kernel)
        / max(np.linalg.norm(reference_kernel), 1.0e-15)
    )


def load_saved_states(
    best_path: Path,
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray]:
    data_path = best_path.with_name("best_encoding_data.npz")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} is required for the target-optimized reference"
        )
    with np.load(data_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["y_dev"], split.y_dev):
            raise RuntimeError(f"{data_path}: development labels differ")
        if not np.array_equal(saved["y_test"], split.y_test):
            raise RuntimeError(f"{data_path}: test labels differ")
        return (
            np.asarray(saved["dev_states"], dtype=np.complex128),
            np.asarray(saved["test_states"], dtype=np.complex128),
        )


def transferred_states(
    encoder,
    split: DatasetSplit,
    seed: int,
    state_dir: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = state_dir / f"spirals_seed_{seed:04d}_transferred_states.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if not np.array_equal(cached["y_dev"], split.y_dev):
                raise RuntimeError(f"{cache_path}: cached development labels differ")
            if not np.array_equal(cached["y_test"], split.y_test):
                raise RuntimeError(f"{cache_path}: cached test labels differ")
            return (
                np.asarray(cached["dev_states"], dtype=np.complex128),
                np.asarray(cached["test_states"], dtype=np.complex128),
            )

    state_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"transfer_seed_{seed:04d}_",
        dir=str(state_dir),
    ) as temporary:
        workdir = Path(temporary)
        dev_states = encoder.encode_dataset(
            split.X_dev,
            workdir,
            f"spirals_seed_{seed:04d}_dev",
        )
        test_states = encoder.encode_dataset(
            split.X_test,
            workdir,
            f"spirals_seed_{seed:04d}_test",
        )

    np.savez_compressed(
        cache_path,
        dev_states=dev_states,
        test_states=test_states,
        y_dev=split.y_dev,
        y_test=split.y_test,
    )
    return dev_states, test_states


def validate_reconstruction(
    best_path: Path,
    summary: Mapping[str, object],
    split: DatasetSplit,
    evaluation: MapEvaluation,
) -> float | None:
    data_path = best_path.with_name("best_encoding_data.npz")
    if not data_path.exists():
        return None

    with np.load(data_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["y_dev"], split.y_dev):
            raise RuntimeError(f"{data_path}: reconstructed development labels differ")
        if not np.array_equal(saved["y_test"], split.y_test):
            raise RuntimeError(f"{data_path}: reconstructed test labels differ")
        reference = np.asarray(saved["K_dev"], dtype=np.float64)

    relative_error = float(
        np.linalg.norm(evaluation.K_dev - reference)
        / max(np.linalg.norm(reference), 1.0e-15)
    )
    if relative_error > 1.0e-2:
        raise RuntimeError(
            f"{best_path}: analytic reconstruction error {relative_error:.3e}"
        )

    recorded = float(summary["held_out_test_accuracy"])
    if not math.isclose(evaluation.test_accuracy, recorded, abs_tol=1.0e-12):
        raise RuntimeError(
            f"{best_path}: reconstructed accuracy {evaluation.test_accuracy:.6f} "
            f"does not match recorded accuracy {recorded:.6f}"
        )
    return relative_error


def discover_target_maps(
    root: Path,
    family: str,
) -> dict[int, tuple[Path, dict]]:
    records: dict[int, tuple[Path, dict]] = {}
    pattern = f"seed_*/{family}/de/best_encoding.json"
    for path in sorted(root.glob(pattern)):
        summary = load_summary(path)
        seed = int(summary["seed"])
        if seed in records:
            raise ValueError(f"duplicate target seed {seed}")
        records[seed] = (path, summary)
    if not records:
        raise ValueError(f"no target maps found under {root} with {pattern}")
    return records


def make_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_size < 50:
        raise ValueError("--grid-size must be at least 50")
    coordinates = np.linspace(-1.05, 1.05, grid_size)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, points


def accuracy_stats(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def draw_panel(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    decision: np.ndarray,
    split: DatasetSplit,
    title: str,
    panel_label: str,
) -> None:
    zz = decision.reshape(xx.shape)
    ax.contourf(
        xx,
        yy,
        zz,
        levels=[-np.inf, 0.0, np.inf],
        colors=[REGION_BLUE, REGION_RED],
        alpha=0.82,
        antialiased=True,
    )
    if float(np.min(zz)) < 0.0 < float(np.max(zz)):
        ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=0.85)

    marker_by_class = {0: "o", 1: "^"}
    color_by_class = {0: BLUE, 1: RED}
    for label in (0, 1):
        dev_mask = split.y_dev == label
        test_mask = split.y_test == label
        ax.scatter(
            split.X_dev[dev_mask, 0],
            split.X_dev[dev_mask, 1],
            s=7.0,
            marker=marker_by_class[label],
            c=color_by_class[label],
            edgecolors="white",
            linewidths=0.18,
            alpha=0.56,
            zorder=3,
        )
        ax.scatter(
            split.X_test[test_mask, 0],
            split.X_test[test_mask, 1],
            s=18.0,
            marker=marker_by_class[label],
            c=color_by_class[label],
            edgecolors="black",
            linewidths=0.42,
            alpha=0.96,
            zorder=4,
        )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.grid(True, color="black", alpha=0.12, linewidth=0.35, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$x_0$")
    ax.set_title(title, pad=3.0, fontweight="normal")
    ax.text(
        0.025,
        0.965,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.82},
        zorder=6,
    )


def make_figure(
    xx: np.ndarray,
    yy: np.ndarray,
    split: DatasetSplit,
    transfer_decision: np.ndarray,
    optimized_decision: np.ndarray,
    transfer_stats: Mapping[str, float],
    optimized_stats: Mapping[str, float],
    output_prefix: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.18), sharex=True, sharey=True)
    titles = (
        "Frozen moons pulse map\n"
        f"Accuracy = {100.0 * transfer_stats['mean']:.1f} "
        f"$\\pm$ {100.0 * transfer_stats['sample_std']:.1f}%",
        "Spiral-optimized pulse map\n"
        f"Accuracy = {100.0 * optimized_stats['mean']:.1f} "
        f"$\\pm$ {100.0 * optimized_stats['sample_std']:.1f}%",
    )
    for index, (ax, decision, title) in enumerate(
        zip(axes, (transfer_decision, optimized_decision), titles)
    ):
        draw_panel(
            ax,
            xx,
            yy,
            decision,
            split,
            title,
            f"({chr(ord('a') + index)})",
        )
    axes[0].set_ylabel(r"$x_1$")

    legend = (
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE,
               markeredgecolor="none", markersize=4.0, label="Class 0"),
        Line2D([], [], marker="^", linestyle="none", markerfacecolor=RED,
               markeredgecolor="none", markersize=4.0, label="Class 1"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4.4, label="Held-out test point"),
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.83, bottom=0.22, wspace=0.16)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_prefix.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.svm_c <= 0.0:
        raise ValueError("--svm-c must be > 0")
    if args.integration_intervals < 4:
        raise ValueError("--integration-intervals must be >= 4")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if not args.analytic_only and args.fluxpulse is None:
        raise ValueError("--fluxpulse is required unless --analytic-only is used")

    args.outdir.mkdir(parents=True, exist_ok=True)

    source_summary = load_summary(args.source_best)
    if str(source_summary["dataset"]) != "moons":
        raise ValueError("--source-best must describe the moons benchmark")

    target_records = discover_target_maps(args.target_root, args.target_family)
    if any(str(summary["dataset"]) != "spirals" for _, summary in target_records.values()):
        raise ValueError("every target result must describe the spirals benchmark")

    display_seed = (
        int(args.display_seed)
        if args.display_seed is not None
        else int(source_summary["seed"])
    )
    if display_seed not in target_records:
        raise ValueError(f"display seed {display_seed} is not available")

    source_split = reconstruct_split(source_summary)
    source_dev_states, source_test_states = load_saved_states(
        args.source_best,
        source_split,
    )
    source_evaluation = evaluate_state_arrays(
        source_split,
        source_dev_states,
        source_test_states,
        source_summary["best_params"],
        args.svm_c,
    )
    if not math.isclose(
        source_evaluation.test_accuracy,
        float(source_summary["held_out_test_accuracy"]),
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("stored source states do not reproduce source accuracy")
    source_reconstruction_error = analytic_kernel_error(
        source_split,
        source_summary["best_params"],
        source_evaluation.K_dev,
    )

    source_tau = float(source_summary["best_params"]["physical"]["pulse_duration"])
    source_template_path = Path(str(source_summary.get("template", "")))
    encoder = None
    if not args.analytic_only:
        helper_path = (
            args.fluxpulse_helper
            if args.fluxpulse_helper is not None
            else Path(__file__).with_name("qml_time_convergence.py")
        )
        helper = load_fluxpulse_helper(helper_path)
        source_params = helper.params_from_summary(source_summary)
        source_tau = helper.pulse_duration_from_summary(source_summary)
        source_template_path = helper.resolve_template_path(source_summary, None)
        with source_template_path.open("r", encoding="utf-8") as stream:
            source_template = json.load(stream)
        encoder = helper.FluxPulseEncoder(
            fluxpulse_binary=args.fluxpulse,
            template=source_template,
            params=source_params,
            pulse_duration=source_tau,
            integration_intervals=args.integration_intervals,
            jobs=args.jobs,
        )

    xx, yy, grid_points = make_grid(args.grid_size)
    per_seed: list[dict] = []
    display_split: DatasetSplit | None = None
    display_transfer: MapEvaluation | None = None
    display_optimized: MapEvaluation | None = None

    for seed in sorted(target_records):
        target_path, target_summary = target_records[seed]
        split = reconstruct_split(target_summary)

        if args.analytic_only:
            transferred_dev_states = encode_states(
                split.X_dev,
                source_summary["best_params"],
            )
            transferred_test_states = encode_states(
                split.X_test,
                source_summary["best_params"],
            )
        else:
            transferred_dev_states, transferred_test_states = transferred_states(
                encoder,
                split,
                seed,
                args.outdir / "transferred_states",
                args.force,
            )
        transfer = evaluate_state_arrays(
            split,
            transferred_dev_states,
            transferred_test_states,
            source_summary["best_params"],
            args.svm_c,
        )

        optimized_dev_states, optimized_test_states = load_saved_states(
            target_path,
            split,
        )
        optimized = evaluate_state_arrays(
            split,
            optimized_dev_states,
            optimized_test_states,
            target_summary["best_params"],
            args.svm_c,
        )
        recorded_target_accuracy = float(
            target_summary["held_out_test_accuracy"]
        )
        if not math.isclose(
            optimized.test_accuracy,
            recorded_target_accuracy,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"{target_path}: stored states reproduce accuracy "
                f"{optimized.test_accuracy:.6f}, expected "
                f"{recorded_target_accuracy:.6f}"
            )

        transfer_reconstruction_error = analytic_kernel_error(
            split,
            source_summary["best_params"],
            transfer.K_dev,
        )
        target_reconstruction_error = analytic_kernel_error(
            split,
            target_summary["best_params"],
            optimized.K_dev,
        )
        if max(transfer_reconstruction_error, target_reconstruction_error) > 1.0e-2:
            raise RuntimeError(
                f"seed {seed}: analytic boundary reconstruction exceeds 1e-2"
            )

        per_seed.append(
            {
                "target_seed": int(seed),
                "transferred_test_accuracy": float(transfer.test_accuracy),
                "target_optimized_test_accuracy": float(optimized.test_accuracy),
                "transferred_minus_optimized": float(
                    transfer.test_accuracy - optimized.test_accuracy
                ),
                "changed_predictions_between_maps": int(
                    np.sum(transfer.test_prediction != optimized.test_prediction)
                ),
                "transferred_map_boundary_relative_kernel_error": float(
                    transfer_reconstruction_error
                ),
                "target_map_boundary_relative_kernel_error": float(
                    target_reconstruction_error
                ),
            }
        )

        print(
            f"[transfer] target_seed={seed} "
            f"frozen_moons={transfer.test_accuracy:.4f} "
            f"spiral_optimized={optimized.test_accuracy:.4f}"
        )

        if seed == display_seed:
            display_split = split
            # Use one internally consistent analytic reconstruction for the
            # dense boundary grid. Reported accuracy statistics still come
            # from the FluxPulse states evaluated above.
            display_transfer = evaluate_map(
                split,
                source_summary["best_params"],
                args.svm_c,
                grid_points,
            )
            display_optimized = evaluate_map(
                split,
                target_summary["best_params"],
                args.svm_c,
                grid_points,
            )

    if display_split is None or display_transfer is None or display_optimized is None:
        raise RuntimeError("failed to construct the displayed target boundary")
    if display_transfer.decision is None or display_optimized.decision is None:
        raise RuntimeError("displayed decision values were not computed")

    transfer_values = [row["transferred_test_accuracy"] for row in per_seed]
    optimized_values = [row["target_optimized_test_accuracy"] for row in per_seed]
    differences = [row["transferred_minus_optimized"] for row in per_seed]
    transfer_stats = accuracy_stats(transfer_values)
    optimized_stats = accuracy_stats(optimized_values)
    difference_stats = accuracy_stats(differences)

    output_prefix = args.outdir / "moons-to-spirals-transfer-boundaries"
    configure_matplotlib()
    make_figure(
        xx,
        yy,
        display_split,
        display_transfer.decision,
        display_optimized.decision,
        transfer_stats,
        optimized_stats,
        output_prefix,
        args.dpi,
    )

    np.savez_compressed(
        args.outdir / "moons-to-spirals-transfer-boundary-data.npz",
        xx=xx,
        yy=yy,
        X_dev=display_split.X_dev,
        X_test=display_split.X_test,
        y_dev=display_split.y_dev,
        y_test=display_split.y_test,
        transferred_decision=display_transfer.decision.reshape(xx.shape),
        target_optimized_decision=display_optimized.decision.reshape(xx.shape),
        transferred_test_prediction=display_transfer.test_prediction,
        target_optimized_test_prediction=display_optimized.test_prediction,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "interpretation": (
            "The pulse-map parameters are frozen after moons optimization; "
            "only the precomputed-kernel SVM is fitted on each spiral split."
        ),
        "source": {
            "best_encoding": str(args.source_best),
            "dataset": source_summary["dataset"],
            "seed": int(source_summary["seed"]),
            "recorded_source_test_accuracy": float(
                source_summary["held_out_test_accuracy"]
            ),
            "reconstructed_source_test_accuracy": float(
                source_evaluation.test_accuracy
            ),
            "boundary_reconstruction_relative_kernel_error": float(
                source_reconstruction_error
            ),
            "pulse_duration": float(source_tau),
            "template": str(source_template_path),
        },
        "target": {
            "root": str(args.target_root),
            "family": args.target_family,
            "display_seed": int(display_seed),
            "svm_c": float(args.svm_c),
            "transfer_backend": (
                "analytic_exponentiation" if args.analytic_only else "fluxpulse"
            ),
            "integration_intervals": int(args.integration_intervals),
            "transfer_accuracy": transfer_stats,
            "target_optimized_accuracy": optimized_stats,
            "transfer_minus_optimized_accuracy": difference_stats,
            "per_seed": per_seed,
        },
        "outputs": {
            "pdf": str(output_prefix.with_suffix(".pdf")),
            "png": str(output_prefix.with_suffix(".png")),
            "svg": str(output_prefix.with_suffix(".svg")),
            "boundary_data": str(
                args.outdir / "moons-to-spirals-transfer-boundary-data.npz"
            ),
        },
    }
    summary_path = args.outdir / "moons-to-spirals-transfer-summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[transfer] wrote {summary_path}")


if __name__ == "__main__":
    main()
