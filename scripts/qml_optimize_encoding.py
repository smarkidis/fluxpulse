#!/usr/bin/env python3
"""
FluxPulse QML pulse-encoding optimizer.

Version 9 keeps the downstream objective introduced in Version 5 and removes
the remaining exact redundancies from the one-segment feature map. It also
keeps the reference Hamiltonian fixed during optimization and uses reflective
boundary handling for all search operators. It includes moons, concentric
circles, and a configurable two-arm spiral as two-dimensional nonlinear
benchmarks.

Main methodological changes
============================

1. Product-map redundancy reduction
-----------------------------------
For an uncoupled product template, the one-qubit fidelity kernel generated
from |0> is invariant to the XY-plane phase phi_q and to the sign of the
data-dependent detuning delta_q. Therefore the product search no longer wastes
dimensions on these exact symmetries.

For each product qubit q, the search variables are now only

    alpha_q >= 0
    |delta_q| >= 0

where

    alpha_q = A_q * tau
    delta_q = Delta_q * tau

and tau is a fixed physical pulse duration.

Thus a 2-qubit product search has 4 effective parameters instead of the
nominal 6:

    (alpha_0, |delta_0|, alpha_1, |delta_1|).

The physical phase is fixed to phi_q = 0 for product templates.

For a template containing fixed ZZ edges, the azimuthal drive phases remain
kernel-invariant: local Z-frame rotations commute with ZZ and leave the
initial computational state invariant up to phase. The interacting search is
therefore reduced to

    (alpha_q, delta_q) for every qubit
    plus the common pulse duration tau.

The signed detunings are retained because the ZZ interaction breaks their
product-map sign symmetry. A 2-qubit ZZ search has 5 parameters.

The ZZ strengths are read from the template and never changed by a candidate.
For an edge with fixed strength zeta, the resulting interaction angle is

    chi = zeta * tau.

This treats coupling strength as a property of the reference system and pulse
duration as an experimental control.

2. Downstream QSVM fitness
--------------------------
The held-out test set is removed before optimization and is never touched
while candidates are selected.

For every candidate:
    * encode the complete development set,
    * build the fidelity kernel,
    * evaluate a fixed StratifiedKFold partition,
    * train SVC(kernel="precomputed") on each fold,
    * calculate the validation hinge loss from the decision margin.

The optimization fitness is

    fitness = - mean_cross_validation_hinge_loss.

This gives a much smoother downstream objective than classification accuracy.

Cross-validation accuracy is reported as the understandable performance
measure. Centered kernel-target alignment is retained only as a diagnostic.

3. Feature-map diagnostics
--------------------------
For every final selected candidate, the script reports:

    * kernel eigenvalue spectrum,
    * numerical rank at a relative threshold,
    * entropy effective rank,
    * participation ratio,
    * delta_q -> -delta_q invariance probes.
    * active search-bound diagnostics.

For an ideal uncoupled product template, detuning-sign kernel differences
should be at numerical roundoff. For an interacting ZZ template, they need
not vanish.

4. Optimizer benchmark retained
--------------------------------
The same candidate-evaluation budget can be used with:

    --optimizer ga
    --optimizer random
    --optimizer de
    --optimizer all

All optimizers use the same fixed cross-validation folds and candidate budget.

5. Reflective boundary handling
-------------------------------
Search proposals outside a parameter interval are reflected back into that
interval. Reflection preserves the size of an attempted optimizer step and
avoids the artificial accumulation of candidates at a bound caused by
clipping. Repeated reflection handles proposals that overshoot an interval by
more than its width.

Datasets
========
    moons
    circles
    spirals
    correlated4
    breast_cancer_4f

No PCA is used.

Example
=======

python3 scripts/qml_optimize_encoding.py \
  --dataset moons \
  --template configs/qml/2q_ideal_encoding.json \
  --compare-template configs/qml/2q_entangling_encoding.json \
  --fluxpulse ./build/fluxpulse_run \
  --samples 200 \
  --optimizer all \
  --population 12 \
  --max-evals 72 \
  --cv-folds 5 \
  --num-seeds 1 \
  --seed 7 \
  --reference-zz-strength -100.0 \
  --duration-min 0.005 \
  --duration-max 0.030 \
  --outdir out/qml/moons_qsvm_hinge_v6
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.datasets import load_breast_cancer, make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_VERSION = "2026-08-11-spirals-reflective-boundaries-v9"

# Lower bound for the squared radial coordinate used to sample each spiral
# arm. Keeping a few points away from the singular centre makes the benchmark
# reproducible without introducing a user-facing parameter that would be hard
# to interpret physically.
SPIRAL_RADIAL_FLOOR = 0.03

BREAST_CANCER_FEATURES = (
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean concavity",
)

OPTIMIZER_ORDER = ("ga", "random", "de")


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class EncodingParams:
    alpha: Tuple[float, ...]
    delta: Tuple[float, ...]
    duration: float
    interacting: bool

    def flat_key(self, ndigits: int = 12) -> Tuple[float, ...]:
        values: List[float] = []
        for q in range(len(self.alpha)):
            values.extend(
                (
                    round(self.alpha[q], ndigits),
                    round(self.delta[q], ndigits),
                )
            )
        if self.interacting:
            values.append(round(self.duration, ndigits))
        return tuple(values)

    def to_dict(self, zz_edges: Optional[Sequence[dict]] = None) -> dict:
        derived_angles = []
        for edge in zz_edges or ():
            item = copy.deepcopy(edge)
            item["strength"] = float(edge["strength"])
            item["angle"] = float(edge["strength"] * self.duration)
            derived_angles.append(item)

        return {
            "dimensionless": {
                "alpha": [float(v) for v in self.alpha],
                "delta": [float(v) for v in self.delta],
            },
            "physical": {
                "pulse_duration": float(self.duration),
                "amplitude": [
                    float(v / self.duration)
                    for v in self.alpha
                ],
                "detuning": [
                    float(v / self.duration)
                    for v in self.delta
                ],
                "reference_zz_edges": derived_angles,
            },
        }


@dataclass(frozen=True)
class Bounds:
    alpha: Tuple[float, float]
    delta: Tuple[float, float]
    duration: Tuple[float, float]

    def product_delta_abs_max(self) -> float:
        return max(abs(self.delta[0]), abs(self.delta[1]))

    def matrix(
        self,
        nqubits: int,
        optimize_zz: bool,
    ) -> np.ndarray:
        rows: List[Tuple[float, float]] = []

        if not optimize_zz:
            # Remove exact product-map symmetries:
            # phase is fixed to zero and detuning is represented by |delta|.
            dmax = self.product_delta_abs_max()
            for _ in range(nqubits):
                rows.append(self.alpha)
                rows.append((0.0, dmax))
            return np.asarray(rows, dtype=np.float64)

        # Interacting template: phase is also invariant for a one-segment
        # X/Y drive with fixed ZZ coupling. Keep signed detuning and optimize
        # duration so the fixed interaction accumulates a variable angle.
        for _ in range(nqubits):
            rows.append(self.alpha)
            rows.append(self.delta)

        rows.append(self.duration)
        return np.asarray(rows, dtype=np.float64)

    def names(
        self,
        nqubits: int,
        optimize_zz: bool,
    ) -> List[str]:
        names: List[str] = []

        if not optimize_zz:
            for q in range(nqubits):
                names.extend(
                    (
                        f"alpha_{q}",
                        f"abs_delta_{q}",
                    )
                )
            return names

        for q in range(nqubits):
            names.extend(
                (
                    f"alpha_{q}",
                    f"delta_{q}",
                )
            )
        names.append("pulse_duration")
        return names


@dataclass
class DatasetSplit:
    X_dev: np.ndarray
    X_test: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class CandidateMetrics:
    fitness: float
    hinge_loss: float
    cv_accuracy: float
    alignment: float


@dataclass
class SearchResult:
    optimizer: str
    best_vector: np.ndarray
    best_metrics: CandidateMetrics
    history: List[dict]
    objective_evaluations: int
    unique_candidate_evaluations: int


# =============================================================================
# Dataset generation and preprocessing: no PCA
# =============================================================================

def load_raw_dataset(
    dataset: str,
    seed: int,
    samples: int,
    moons_noise: float,
    circles_noise: float,
    circles_factor: float,
    spiral_noise: float,
    spiral_turns: float,
    correlation: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:

    if dataset == "moons":
        if samples < 20:
            raise ValueError("--samples must be >= 20 for moons")

        X, y = make_moons(
            n_samples=samples,
            noise=moons_noise,
            random_state=seed,
        )

        return (
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.int64),
            {
                "dataset": "moons",
                "features": ["moon_x0", "moon_x1"],
                "samples": int(samples),
                "moons_noise": float(moons_noise),
            },
        )

    if dataset == "circles":
        if samples < 20:
            raise ValueError("--samples must be >= 20 for circles")
        if circles_noise < 0.0:
            raise ValueError("--circles-noise must be >= 0")
        if not (0.0 < circles_factor < 1.0):
            raise ValueError("--circles-factor must lie in (0,1)")

        X, y = make_circles(
            n_samples=samples,
            noise=circles_noise,
            factor=circles_factor,
            random_state=seed,
        )

        return (
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.int64),
            {
                "dataset": "circles",
                "features": ["circle_x0", "circle_x1"],
                "samples": int(samples),
                "circles_noise": float(circles_noise),
                "circles_factor": float(circles_factor),
                "structure": "concentric_circles",
            },
        )

    if dataset == "spirals":
        if samples < 20:
            raise ValueError("--samples must be >= 20 for spirals")
        if not math.isfinite(spiral_noise) or spiral_noise < 0.0:
            raise ValueError("--spiral-noise must be finite and >= 0")
        if not math.isfinite(spiral_turns) or spiral_turns <= 0.0:
            raise ValueError("--spiral-turns must be finite and > 0")

        rng = np.random.default_rng(seed)
        class_sizes = (samples // 2, samples - samples // 2)

        def make_spiral_arm(n_points: int, phase: float) -> np.ndarray:
            # Uniform sampling in r**2 gives approximately uniform point
            # density over the disk instead of crowding samples at the centre.
            radius = np.sqrt(
                rng.uniform(
                    SPIRAL_RADIAL_FLOOR,
                    1.0,
                    size=n_points,
                )
            )
            angle = 2.0 * math.pi * spiral_turns * radius + phase
            arm = np.column_stack(
                (
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                )
            )
            if spiral_noise > 0.0:
                arm += rng.normal(
                    loc=0.0,
                    scale=spiral_noise,
                    size=arm.shape,
                )
            return arm

        X = np.vstack(
            (
                make_spiral_arm(class_sizes[0], 0.0),
                make_spiral_arm(class_sizes[1], math.pi),
            )
        )
        y = np.concatenate(
            (
                np.zeros(class_sizes[0], dtype=np.int64),
                np.ones(class_sizes[1], dtype=np.int64),
            )
        )

        # Avoid returning samples ordered by class. The downstream split is
        # stratified, but shuffling here also makes the raw dataset safer for
        # external callers and diagnostic plots.
        permutation = rng.permutation(samples)
        X = X[permutation]
        y = y[permutation]

        return (
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.int64),
            {
                "dataset": "spirals",
                "features": ["spiral_x0", "spiral_x1"],
                "samples": int(samples),
                "spiral_turns": float(spiral_turns),
                "spiral_noise": float(spiral_noise),
                "spiral_radial_floor": float(SPIRAL_RADIAL_FLOOR),
                "structure": "two_arm_spiral",
                "angle_rule": "theta = 2*pi*turns*r + class*pi",
            },
        )

    if dataset == "correlated4":
        if samples < 20:
            raise ValueError("--samples must be >= 20 for correlated4")
        if not (-0.999 <= correlation <= 0.999):
            raise ValueError("--correlation must lie in [-0.999, 0.999]")

        rng = np.random.default_rng(seed)

        z0 = rng.normal(size=samples)
        z1 = rng.normal(size=samples)
        e2 = rng.normal(size=samples)
        e3 = rng.normal(size=samples)

        rho = float(correlation)
        residual = math.sqrt(max(0.0, 1.0 - rho * rho))

        x0 = z0
        x1 = z1
        x2 = rho * z0 + residual * e2
        x3 = rho * z1 + residual * e3

        X = np.column_stack((x0, x1, x2, x3)).astype(np.float64)

        score = x0 * x1 + x2 * x3
        y = (score > 0.0).astype(np.int64)

        return (
            X,
            y,
            {
                "dataset": "correlated4",
                "features": ["x0", "x1", "x2", "x3"],
                "samples": int(samples),
                "correlation": rho,
                "label_rule": "1[x0*x1 + x2*x3 > 0]",
                "empirical_feature_correlation":
                    np.corrcoef(X, rowvar=False).tolist(),
            },
        )

    if dataset == "breast_cancer_4f":
        data = load_breast_cancer()
        names = list(data.feature_names)

        indices: List[int] = []
        for feature in BREAST_CANCER_FEATURES:
            if feature not in names:
                raise RuntimeError(
                    f"feature {feature!r} not found in sklearn dataset"
                )
            indices.append(names.index(feature))

        X_full = np.asarray(data.data[:, indices], dtype=np.float64)
        y_full = np.asarray(data.target, dtype=np.int64)

        if samples <= 0 or samples >= len(y_full):
            X = X_full
            y = y_full
        else:
            if samples < 20:
                raise ValueError("--samples must be 0 or >= 20")
            X, _, y, _ = train_test_split(
                X_full,
                y_full,
                train_size=samples,
                random_state=seed,
                stratify=y_full,
            )

        return (
            X,
            y,
            {
                "dataset": "breast_cancer_4f",
                "features": list(BREAST_CANCER_FEATURES),
                "samples": int(len(y)),
                "full_dataset_samples": int(len(y_full)),
            },
        )

    raise ValueError(
        f"unsupported dataset {dataset!r}; choose "
        "moons, circles, spirals, correlated4, or breast_cancer_4f"
    )


def make_dataset_split(
    dataset: str,
    seed: int,
    samples: int,
    test_size: float,
    moons_noise: float,
    circles_noise: float,
    circles_factor: float,
    spiral_noise: float,
    spiral_turns: float,
    correlation: float,
) -> DatasetSplit:

    X, y, metadata = load_raw_dataset(
        dataset=dataset,
        seed=seed,
        samples=samples,
        moons_noise=moons_noise,
        circles_noise=circles_noise,
        circles_factor=circles_factor,
        spiral_noise=spiral_noise,
        spiral_turns=spiral_turns,
        correlation=correlation,
    )

    # The test set is removed before any candidate optimization.
    X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # No PCA. The development-set scaler is fixed before the pulse search.
    # Test data are transformed with development statistics only.
    scaler = StandardScaler()
    X_dev_std = scaler.fit_transform(X_dev_raw)
    X_test_std = scaler.transform(X_test_raw)

    maxabs = np.max(np.abs(X_dev_std), axis=0)
    maxabs[maxabs < 1.0e-12] = 1.0

    def scale(z: np.ndarray) -> np.ndarray:
        return np.clip(z / maxabs, -1.0, 1.0)

    metadata = dict(metadata)
    metadata.update(
        {
            "seed": int(seed),
            "preprocessing": (
                "StandardScaler(development) -> development maxabs scaling "
                "-> clip [-1,1]"
            ),
            "pca": False,
            "development_samples": int(len(y_dev)),
            "test_samples": int(len(y_test)),
            "class_counts_all": {
                str(int(c)): int(np.sum(y == c))
                for c in np.unique(y)
            },
            "development_maxabs_after_standardization": maxabs.tolist(),
        }
    )

    return DatasetSplit(
        X_dev=scale(X_dev_std),
        X_test=scale(X_test_std),
        y_dev=np.asarray(y_dev, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        metadata=metadata,
    )


def make_fixed_cv_folds(
    y_dev: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:

    if n_splits < 2:
        raise ValueError("--cv-folds must be >= 2")

    classes, counts = np.unique(y_dev, return_counts=True)
    if len(classes) != 2:
        raise ValueError("this workflow expects binary labels")

    if np.min(counts) < n_splits:
        raise ValueError(
            f"--cv-folds={n_splits} exceeds the smallest class count "
            f"({int(np.min(counts))}) in the development set"
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    dummy = np.zeros((len(y_dev), 1), dtype=np.float64)

    return [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
        )
        for train_idx, val_idx in splitter.split(dummy, y_dev)
    ]


# =============================================================================
# FluxPulse model/config helpers
# =============================================================================

def infer_nqubits(template: dict) -> int:
    model = template.get("model", {})
    model_type = model.get("type", "")

    if model_type == "generic_qubit":
        return 1

    if model_type in ("coupled_qubits", "duffing_transmon"):
        default_nq = 1 if model_type == "duffing_transmon" else 0
        nq = int(model.get("nqubits", default_nq))
        if nq <= 0:
            raise ValueError(f"{model_type} template requires nqubits > 0")
        return nq

    raise ValueError(
        f"cannot infer nqubits from model type {model_type!r}"
    )


def template_zz_edges(template: dict) -> List[dict]:
    zz = template.get("model", {}).get("zz", [])
    if zz is None:
        return []
    if not isinstance(zz, list):
        raise ValueError("model.zz must be a list")
    return zz


def template_optimizes_zz(template: dict) -> bool:
    return len(template_zz_edges(template)) > 0


def configure_reference_zz_strength(
    template: dict,
    override: Optional[float],
) -> None:
    """Set an optional run-wide reference value, then validate fixed edges."""
    edges = template_zz_edges(template)

    if override is not None:
        if not edges:
            raise ValueError(
                "--reference-zz-strength was supplied but the template "
                "contains no model.zz edges"
            )
        if not math.isfinite(override) or abs(override) <= 1.0e-15:
            raise ValueError(
                "--reference-zz-strength must be finite and nonzero"
            )
        for edge in edges:
            edge["strength"] = float(override)

    for edge_id, edge in enumerate(edges):
        if "strength" not in edge:
            raise ValueError(
                f"model.zz[{edge_id}] requires a fixed strength; set it "
                "in the template or use --reference-zz-strength"
            )
        strength = float(edge["strength"])
        if not math.isfinite(strength) or abs(strength) <= 1.0e-15:
            raise ValueError(
                f"model.zz[{edge_id}].strength must be finite and nonzero; "
                "the optimizer no longer changes hardware coupling strengths"
            )


def force_ideal_model(cfg: dict, nqubits: int) -> None:
    model = cfg["model"]

    if nqubits == 1 and model.get("type") == "generic_qubit":
        model["t1"] = 0.0
        model["tphi"] = 0.0
    else:
        model["t1"] = [0.0] * nqubits
        model["tphi"] = [0.0] * nqubits

    model["enable_drive_x"] = True
    model["enable_drive_y"] = False


def set_data_detuning(
    cfg: dict,
    x: np.ndarray,
    params: EncodingParams,
) -> None:

    model = cfg["model"]
    nq = len(x)

    detuning = [
        float((params.delta[q] / params.duration) * x[q])
        for q in range(nq)
    ]

    if nq == 1 and model.get("type") == "generic_qubit":
        model["detuning"] = detuning[0]
    else:
        model["detuning"] = detuning


def build_candidate_config(
    template: dict,
    x: np.ndarray,
    params: EncodingParams,
    result_path: Path,
    integration_steps: int,
) -> dict:

    cfg = copy.deepcopy(template)
    nq = infer_nqubits(cfg)

    if len(x) != nq:
        raise ValueError(
            f"sample has {len(x)} features but template has {nq} qubits"
        )

    optimize_zz = template_optimizes_zz(cfg)
    if optimize_zz != params.interacting:
        raise ValueError("parameter/template ZZ mismatch")

    # Couplings are properties of the reference model. A candidate is never
    # allowed to alter them; interacting candidates change only duration.
    configure_reference_zz_strength(cfg, override=None)

    force_ideal_model(cfg, nq)
    set_data_detuning(
        cfg,
        x,
        params,
    )

    cfg.setdefault("run", {})
    cfg["run"]["trajectories"] = 1
    cfg["run"]["renormalize_each_step"] = True
    cfg["run"]["sample_stride"] = int(integration_steps)

    cfg.setdefault("output", {})
    cfg["output"]["result_path"] = str(result_path)
    cfg["output"]["write_trace"] = False

    cfg.setdefault("trace", {})
    cfg["trace"]["emit_segments"] = False
    cfg["trace"]["emit_counters"] = False

    dt = float(params.duration) / float(integration_steps)

    cfg.setdefault("program", {})
    cfg["program"]["grid"] = {
        "type": "uniform",
        "t0": 0.0,
        "dt": dt,
        "n_points": int(integration_steps + 1),
    }

    pulses: List[dict] = []

    for q in range(nq):
        amplitude = float(params.alpha[q] / params.duration)
        ux = float(amplitude * x[q])

        pulses.append(
            {
                "channel": f"drive_x{q}",
                "envelope": {
                    "type": "piecewise_constant",
                    "segments": [
                        {
                            "t0": 0.0,
                            "t1": float(params.duration),
                            "value": ux,
                        }
                    ],
                },
            }
        )

    cfg["program"]["pulses"] = pulses
    return cfg


# =============================================================================
# FluxPulse execution
# =============================================================================

def load_final_state(
    result_path: Path,
    expected_dim: int,
) -> np.ndarray:

    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    fs = result.get("final_state")
    if fs is None:
        raise RuntimeError(
            f"{result_path}: result does not contain final_state"
        )

    re = np.asarray(fs.get("real", []), dtype=np.float64)
    im = np.asarray(fs.get("imag", []), dtype=np.float64)

    if len(re) != expected_dim or len(im) != expected_dim:
        raise RuntimeError(
            f"{result_path}: final-state dimension mismatch; "
            f"got {len(re)}, expected {expected_dim}"
        )

    psi = re + 1j * im
    norm2 = float(np.vdot(psi, psi).real)

    if not np.isfinite(norm2) or norm2 <= 0.0:
        raise RuntimeError(
            f"{result_path}: invalid final-state norm2={norm2}"
        )

    psi /= math.sqrt(norm2)
    return psi


class FluxPulseEncoder:
    def __init__(
        self,
        fluxpulse_binary: Path,
        template: dict,
        pulse_duration: float,
        integration_steps: int,
        jobs: int,
    ):
        self.binary = fluxpulse_binary.resolve()
        self.template = template
        self.nqubits = infer_nqubits(template)
        self.dim = 1 << self.nqubits
        self.optimize_zz = template_optimizes_zz(template)
        self.zz_edges = copy.deepcopy(template_zz_edges(template))
        self.pulse_duration = float(pulse_duration)
        self.integration_steps = int(integration_steps)
        self.jobs = max(1, int(jobs))

        if not self.binary.exists():
            raise FileNotFoundError(
                f"FluxPulse executable not found: {self.binary}"
            )

    def encode_one(
        self,
        x: np.ndarray,
        params: EncodingParams,
        workdir: Path,
        tag: str,
    ) -> np.ndarray:

        cfg_path = workdir / f"{tag}.json"
        result_path = workdir / f"{tag}_result.json"

        cfg = build_candidate_config(
            template=self.template,
            x=x,
            params=params,
            result_path=result_path,
            integration_steps=self.integration_steps,
        )

        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        proc = subprocess.run(
            [str(self.binary), str(cfg_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"FluxPulse failed for {cfg_path}\n"
                f"return code: {proc.returncode}\n"
                f"stdout:\n{proc.stdout[-4000:]}\n"
                f"stderr:\n{proc.stderr[-4000:]}"
            )

        psi = load_final_state(
            result_path=result_path,
            expected_dim=self.dim,
        )

        try:
            cfg_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
        except OSError:
            pass

        return psi

    def encode_dataset(
        self,
        X: np.ndarray,
        params: EncodingParams,
        workdir: Path,
        prefix: str,
    ) -> np.ndarray:

        states = np.empty(
            (len(X), self.dim),
            dtype=np.complex128,
        )

        if self.jobs == 1:
            for i, x in enumerate(X):
                states[i] = self.encode_one(
                    x=x,
                    params=params,
                    workdir=workdir,
                    tag=f"{prefix}_{i:05d}",
                )
            return states

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = {
                pool.submit(
                    self.encode_one,
                    X[i],
                    params,
                    workdir,
                    f"{prefix}_{i:05d}",
                ): i
                for i in range(len(X))
            }

            for future in as_completed(futures):
                i = futures[future]
                states[i] = future.result()

        return states


# =============================================================================
# Kernel, downstream CV fitness, alignment, spectrum
# =============================================================================

def fidelity_kernel(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:

    overlap = A.conj() @ B.T
    K = np.abs(overlap) ** 2
    return np.clip(K.real, 0.0, 1.0)


def center_square_matrix(M: np.ndarray) -> np.ndarray:
    row_mean = np.mean(M, axis=1, keepdims=True)
    col_mean = np.mean(M, axis=0, keepdims=True)
    grand_mean = float(np.mean(M))
    return M - row_mean - col_mean + grand_mean


def centered_kernel_alignment(
    K: np.ndarray,
    y: np.ndarray,
) -> float:

    y_pm = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    Y = np.outer(y_pm, y_pm)

    Kc = center_square_matrix(K)
    Yc = center_square_matrix(Y)

    numerator = float(np.sum(Kc * Yc))
    denominator = math.sqrt(
        float(np.sum(Kc * Kc))
        * float(np.sum(Yc * Yc))
    )

    if denominator <= 1.0e-15:
        return 0.0

    return numerator / denominator


def qsvm_cross_validation_metrics(
    K: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    svm_c: float,
) -> Tuple[float, float]:

    losses: List[float] = []
    accuracies: List[float] = []

    for train_idx, val_idx in folds:
        K_train = K[np.ix_(train_idx, train_idx)]
        K_val = K[np.ix_(val_idx, train_idx)]

        clf = SVC(
            kernel="precomputed",
            C=svm_c,
        )
        clf.fit(
            K_train,
            y[train_idx],
        )

        decision = np.asarray(
            clf.decision_function(K_val),
            dtype=np.float64,
        ).reshape(-1)

        # For binary SVC, positive decision values correspond to classes_[1].
        positive_class = clf.classes_[1]
        y_pm = np.where(
            y[val_idx] == positive_class,
            1.0,
            -1.0,
        )

        hinge = np.maximum(
            0.0,
            1.0 - y_pm * decision,
        )

        losses.append(float(np.mean(hinge)))

        prediction = clf.predict(K_val)
        accuracies.append(
            float(
                accuracy_score(
                    y[val_idx],
                    prediction,
                )
            )
        )

    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
    )


def kernel_spectrum(
    K: np.ndarray,
    relative_threshold: float,
    max_reported: int = 16,
) -> dict:

    Ksym = 0.5 * (K + K.T)

    eigvals = np.linalg.eigvalsh(Ksym)
    eigvals = np.sort(eigvals)[::-1]

    # Tiny negative values can arise from roundoff.
    eigvals_nonnegative = np.maximum(eigvals, 0.0)

    largest = (
        float(eigvals_nonnegative[0])
        if len(eigvals_nonnegative)
        else 0.0
    )

    threshold = relative_threshold * largest

    numerical_rank = int(
        np.sum(eigvals_nonnegative > threshold)
    )

    total = float(np.sum(eigvals_nonnegative))

    if total <= 1.0e-15:
        effective_rank_entropy = 0.0
        participation_ratio = 0.0
        normalized = np.zeros_like(eigvals_nonnegative)
    else:
        normalized = eigvals_nonnegative / total
        positive = normalized[normalized > 0.0]

        entropy = -float(
            np.sum(positive * np.log(positive))
        )
        effective_rank_entropy = float(math.exp(entropy))

        denom = float(
            np.sum(eigvals_nonnegative * eigvals_nonnegative)
        )
        participation_ratio = (
            total * total / denom
            if denom > 1.0e-15
            else 0.0
        )

    return {
        "relative_threshold": float(relative_threshold),
        "absolute_threshold": float(threshold),
        "numerical_rank": numerical_rank,
        "effective_rank_entropy": effective_rank_entropy,
        "participation_ratio": float(participation_ratio),
        "largest_eigenvalue": largest,
        "top_eigenvalues": [
            float(v)
            for v in eigvals_nonnegative[:max_reported]
        ],
        "top_normalized_eigenvalues": [
            float(v)
            for v in normalized[:max_reported]
        ],
        "minimum_raw_eigenvalue": (
            float(eigvals[-1])
            if len(eigvals)
            else 0.0
        ),
    }


# =============================================================================
# Search-vector parameterization
# =============================================================================

def vector_to_params(
    vector: Sequence[float],
    nqubits: int,
    optimize_zz: bool,
    fixed_pulse_duration: float,
) -> EncodingParams:

    alpha: List[float] = []
    delta: List[float] = []

    if not optimize_zz:
        expected = 2 * nqubits
        if len(vector) != expected:
            raise ValueError(
                f"product vector length {len(vector)}, expected {expected}"
            )

        k = 0
        for _ in range(nqubits):
            alpha.append(float(max(0.0, vector[k])))
            delta.append(float(abs(vector[k + 1])))
            k += 2

        return EncodingParams(
            alpha=tuple(alpha),
            delta=tuple(delta),
            duration=float(fixed_pulse_duration),
            interacting=False,
        )

    expected = 2 * nqubits + 1
    if len(vector) != expected:
        raise ValueError(
            f"entangling vector length {len(vector)}, expected {expected}"
        )

    k = 0
    for _ in range(nqubits):
        alpha.append(float(max(0.0, vector[k])))
        delta.append(float(vector[k + 1]))
        k += 2

    duration = float(vector[k])

    return EncodingParams(
        alpha=tuple(alpha),
        delta=tuple(delta),
        duration=duration,
        interacting=True,
    )


def reflect_into_interval(
    value: float,
    lower: float,
    upper: float,
) -> float:
    """Reflect a finite scalar into [lower, upper].

    The modulo construction is equivalent to repeatedly reflecting at either
    boundary, but also handles arbitrarily large overshoots in constant time.
    """

    if not math.isfinite(value):
        raise ValueError("cannot repair a non-finite search coordinate")
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("search bounds must be finite")
    if lower >= upper:
        raise ValueError(
            f"invalid reflection interval [{lower}, {upper}]"
        )

    width = upper - lower
    wrapped = math.fmod(value - lower, 2.0 * width)
    if wrapped < 0.0:
        wrapped += 2.0 * width

    if wrapped <= width:
        return float(lower + wrapped)
    return float(upper - (wrapped - width))


def repair_vector(
    vector: np.ndarray,
    bounds: Bounds,
    nqubits: int,
    optimize_zz: bool,
) -> np.ndarray:

    v = np.asarray(vector, dtype=np.float64).copy()
    b = bounds.matrix(nqubits, optimize_zz)

    for j in range(len(v)):
        v[j] = reflect_into_interval(
            value=float(v[j]),
            lower=float(b[j, 0]),
            upper=float(b[j, 1]),
        )

    return v


def sample_population(
    rng: np.random.Generator,
    bounds: Bounds,
    population_size: int,
    nqubits: int,
    optimize_zz: bool,
) -> np.ndarray:

    b = bounds.matrix(
        nqubits,
        optimize_zz,
    )

    return rng.uniform(
        low=b[:, 0],
        high=b[:, 1],
        size=(population_size, b.shape[0]),
    )


def normalized_population_diversity(
    population: np.ndarray,
    bounds: Bounds,
    nqubits: int,
    optimize_zz: bool,
) -> float:

    if len(population) <= 1:
        return 0.0

    b = bounds.matrix(
        nqubits,
        optimize_zz,
    )

    widths = b[:, 1] - b[:, 0]
    widths[widths <= 0.0] = 1.0

    return float(
        np.mean(
            np.std(
                population,
                axis=0,
            )
            / widths
        )
    )


def format_params(
    params: EncodingParams,
    optimize_zz: bool,
    zz_edges: Sequence[dict],
) -> str:

    pieces: List[str] = []

    if not optimize_zz:
        for q in range(len(params.alpha)):
            pieces.append(
                f"q{q}[alpha={params.alpha[q]:.5g},"
                f"|delta|={params.delta[q]:.5g}]"
            )
    else:
        for q in range(len(params.alpha)):
            pieces.append(
                f"q{q}[alpha={params.alpha[q]:.5g},"
                f"delta={params.delta[q]:.5g}]"
            )

        angles = [
            float(edge["strength"] * params.duration)
            for edge in zz_edges
        ]
        pieces.append(f"chi={angles}")

    pieces.append(
        f"tau={params.duration:.5g}"
    )

    return " ".join(pieces)


# =============================================================================
# Candidate evaluator: downstream QSVM CV hinge loss
# =============================================================================

class CandidateEvaluator:
    def __init__(
        self,
        encoder: FluxPulseEncoder,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        folds: Sequence[Tuple[np.ndarray, np.ndarray]],
        svm_c: float,
        workdir: Path,
    ):
        self.encoder = encoder
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.folds = list(folds)
        self.svm_c = float(svm_c)
        self.workdir = workdir

        self.metrics_cache: Dict[
            Tuple[float, ...],
            CandidateMetrics,
        ] = {}

        self.objective_calls = 0
        self.unique_candidate_evaluations = 0

    def evaluate(
        self,
        vector: np.ndarray,
    ) -> CandidateMetrics:

        self.objective_calls += 1

        params = vector_to_params(
            vector,
            self.encoder.nqubits,
            self.encoder.optimize_zz,
            self.encoder.pulse_duration,
        )
        key = params.flat_key()

        if key in self.metrics_cache:
            return self.metrics_cache[key]

        eval_id = self.unique_candidate_evaluations
        self.unique_candidate_evaluations += 1

        states = self.encoder.encode_dataset(
            X=self.X_dev,
            params=params,
            workdir=self.workdir,
            prefix=f"candidate_{eval_id:06d}_dev",
        )

        K = fidelity_kernel(
            states,
            states,
        )

        hinge_loss, cv_accuracy = qsvm_cross_validation_metrics(
            K=K,
            y=self.y_dev,
            folds=self.folds,
            svm_c=self.svm_c,
        )

        alignment = centered_kernel_alignment(
            K,
            self.y_dev,
        )

        metrics = CandidateMetrics(
            fitness=-hinge_loss,
            hinge_loss=hinge_loss,
            cv_accuracy=cv_accuracy,
            alignment=alignment,
        )

        self.metrics_cache[key] = metrics
        return metrics

    def lookup(
        self,
        vector: np.ndarray,
    ) -> CandidateMetrics:

        params = vector_to_params(
            vector,
            self.encoder.nqubits,
            self.encoder.optimize_zz,
            self.encoder.pulse_duration,
        )
        key = params.flat_key()

        if key not in self.metrics_cache:
            raise KeyError(
                "candidate not present in evaluator cache"
            )

        return self.metrics_cache[key]


def evaluate_vector_safely(
    evaluator: CandidateEvaluator,
    vector: np.ndarray,
) -> CandidateMetrics:

    try:
        return evaluator.evaluate(vector)
    except Exception as exc:
        print(
            f"[search] candidate failed: {exc}"
        )
        return CandidateMetrics(
            fitness=-1.0e9,
            hinge_loss=1.0e9,
            cv_accuracy=0.0,
            alignment=-1.0,
        )


def append_history(
    history: List[dict],
    optimizer: str,
    step: int,
    evaluator: CandidateEvaluator,
    encoder: FluxPulseEncoder,
    bounds: Bounds,
    best_vector: np.ndarray,
    best_metrics: CandidateMetrics,
    current_metrics: Sequence[CandidateMetrics],
    current_population: np.ndarray,
) -> None:

    diversity = normalized_population_diversity(
        current_population,
        bounds,
        encoder.nqubits,
        encoder.optimize_zz,
    )

    current_fitness = np.asarray(
        [m.fitness for m in current_metrics],
        dtype=np.float64,
    )

    params = vector_to_params(
        best_vector,
        encoder.nqubits,
        encoder.optimize_zz,
        encoder.pulse_duration,
    )

    record = {
        "step": int(step),
        "objective_evaluations":
            int(evaluator.objective_calls),
        "unique_candidate_evaluations":
            int(evaluator.unique_candidate_evaluations),
        "best_fitness": float(best_metrics.fitness),
        "best_cv_hinge_loss":
            float(best_metrics.hinge_loss),
        "best_cv_accuracy":
            float(best_metrics.cv_accuracy),
        "best_kernel_alignment":
            float(best_metrics.alignment),
        "mean_current_fitness":
            float(np.mean(current_fitness)),
        "std_current_fitness":
            float(np.std(current_fitness)),
        "population_diversity":
            float(diversity),
        "best_params":
            params.to_dict(encoder.zz_edges),
    }
    history.append(record)

    print(
        f"[{optimizer}] "
        f"step={step:02d} "
        f"evals={evaluator.objective_calls:04d} "
        f"fitness={best_metrics.fitness:.6f} "
        f"hinge={best_metrics.hinge_loss:.6f} "
        f"cv_acc={best_metrics.cv_accuracy:.4f} "
        f"align={best_metrics.alignment:.6f} "
        f"mean_fit={np.mean(current_fitness):.6f} "
        f"diversity={diversity:.4f} "
        f"{format_params(params, encoder.optimize_zz, encoder.zz_edges)}"
    )


# =============================================================================
# Optimizers
# =============================================================================


def ga_child(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    bounds: Bounds,
    nqubits: int,
    optimize_zz: bool,
    mutation_probability: float,
    mutation_scale: float,
) -> np.ndarray:

    b = bounds.matrix(
        nqubits,
        optimize_zz,
    )
    child = np.empty_like(parent_a)

    for j in range(len(child)):
        t = float(rng.uniform())

        child[j] = (
            t * parent_a[j]
            + (1.0 - t) * parent_b[j]
        )

        if rng.random() < mutation_probability:
            width = b[j, 1] - b[j, 0]
            child[j] += rng.normal(
                loc=0.0,
                scale=mutation_scale * width,
            )

    return repair_vector(
        child,
        bounds,
        nqubits,
        optimize_zz,
    )


def search_ga(
    evaluator: CandidateEvaluator,
    encoder: FluxPulseEncoder,
    bounds: Bounds,
    population_size: int,
    max_evals: int,
    elite_fraction: float,
    mutation_probability: float,
    mutation_scale: float,
    seed: int,
) -> SearchResult:

    rng = np.random.default_rng(seed)

    population = sample_population(
        rng,
        bounds,
        population_size,
        encoder.nqubits,
        encoder.optimize_zz,
    )

    metrics = [
        evaluate_vector_safely(
            evaluator,
            vector,
        )
        for vector in population
    ]

    best_i = int(
        np.argmax(
            [m.fitness for m in metrics]
        )
    )
    best_vector = population[best_i].copy()
    best_metrics = metrics[best_i]

    history: List[dict] = []

    append_history(
        history,
        "ga",
        0,
        evaluator,
        encoder,
        bounds,
        best_vector,
        best_metrics,
        metrics,
        population,
    )

    elite_count = max(
        1,
        min(
            population_size - 1,
            int(round(elite_fraction * population_size)),
        ),
    )

    step = 1

    while evaluator.objective_calls < max_evals:
        fitness = np.asarray(
            [m.fitness for m in metrics],
            dtype=np.float64,
        )

        order = np.argsort(
            fitness
        )[::-1]

        population = population[order]
        metrics = [
            metrics[i]
            for i in order
        ]

        elites = population[:elite_count].copy()
        elite_metrics = metrics[:elite_count]

        parent_pool_size = max(
            elite_count,
            population_size // 2,
        )
        parent_pool = population[:parent_pool_size]

        available = (
            max_evals
            - evaluator.objective_calls
        )
        normal_children = (
            population_size
            - elite_count
        )
        n_children = min(
            normal_children,
            available,
        )

        children: List[np.ndarray] = []

        for _ in range(n_children):
            ia = int(
                rng.integers(
                    0,
                    len(parent_pool),
                )
            )
            ib = int(
                rng.integers(
                    0,
                    len(parent_pool),
                )
            )

            children.append(
                ga_child(
                    rng=rng,
                    parent_a=parent_pool[ia],
                    parent_b=parent_pool[ib],
                    bounds=bounds,
                    nqubits=encoder.nqubits,
                    optimize_zz=encoder.optimize_zz,
                    mutation_probability=mutation_probability,
                    mutation_scale=mutation_scale,
                )
            )

        child_array = np.asarray(
            children,
            dtype=np.float64,
        )

        child_metrics = [
            evaluate_vector_safely(
                evaluator,
                vector,
            )
            for vector in child_array
        ]

        population = np.vstack(
            (
                elites,
                child_array,
            )
        )
        metrics = (
            list(elite_metrics)
            + child_metrics
        )

        current_i = int(
            np.argmax(
                [m.fitness for m in metrics]
            )
        )

        if (
            metrics[current_i].fitness
            > best_metrics.fitness
        ):
            best_vector = population[current_i].copy()
            best_metrics = metrics[current_i]

        append_history(
            history,
            "ga",
            step,
            evaluator,
            encoder,
            bounds,
            best_vector,
            best_metrics,
            metrics,
            population,
        )

        step += 1

        if n_children < normal_children:
            break

    return SearchResult(
        optimizer="ga",
        best_vector=best_vector,
        best_metrics=best_metrics,
        history=history,
        objective_evaluations=
            int(evaluator.objective_calls),
        unique_candidate_evaluations=
            int(evaluator.unique_candidate_evaluations),
    )


def search_random(
    evaluator: CandidateEvaluator,
    encoder: FluxPulseEncoder,
    bounds: Bounds,
    population_size: int,
    max_evals: int,
    seed: int,
) -> SearchResult:

    rng = np.random.default_rng(seed)

    best_vector: Optional[np.ndarray] = None
    best_metrics = CandidateMetrics(
        fitness=-1.0e300,
        hinge_loss=1.0e300,
        cv_accuracy=0.0,
        alignment=-1.0,
    )

    history: List[dict] = []
    step = 0

    while evaluator.objective_calls < max_evals:
        batch_size = min(
            population_size,
            max_evals - evaluator.objective_calls,
        )

        population = sample_population(
            rng,
            bounds,
            batch_size,
            encoder.nqubits,
            encoder.optimize_zz,
        )

        metrics = [
            evaluate_vector_safely(
                evaluator,
                vector,
            )
            for vector in population
        ]

        current_i = int(
            np.argmax(
                [m.fitness for m in metrics]
            )
        )

        if (
            best_vector is None
            or metrics[current_i].fitness
            > best_metrics.fitness
        ):
            best_vector = population[current_i].copy()
            best_metrics = metrics[current_i]

        assert best_vector is not None

        append_history(
            history,
            "random",
            step,
            evaluator,
            encoder,
            bounds,
            best_vector,
            best_metrics,
            metrics,
            population,
        )

        step += 1

    assert best_vector is not None

    return SearchResult(
        optimizer="random",
        best_vector=best_vector,
        best_metrics=best_metrics,
        history=history,
        objective_evaluations=
            int(evaluator.objective_calls),
        unique_candidate_evaluations=
            int(evaluator.unique_candidate_evaluations),
    )


def de_mutant(
    population: np.ndarray,
    r1: int,
    r2: int,
    r3: int,
    differential_weight: float,
    bounds: Bounds,
    nqubits: int,
    optimize_zz: bool,
) -> np.ndarray:

    base = population[r1].copy()
    diff = (
        population[r2]
        - population[r3]
    )

    mutant = (
        base
        + differential_weight * diff
    )

    return repair_vector(
        mutant,
        bounds,
        nqubits,
        optimize_zz,
    )


def search_de(
    evaluator: CandidateEvaluator,
    encoder: FluxPulseEncoder,
    bounds: Bounds,
    population_size: int,
    max_evals: int,
    differential_weight: float,
    crossover_probability: float,
    seed: int,
) -> SearchResult:

    rng = np.random.default_rng(seed)

    population = sample_population(
        rng,
        bounds,
        population_size,
        encoder.nqubits,
        encoder.optimize_zz,
    )

    metrics = [
        evaluate_vector_safely(
            evaluator,
            vector,
        )
        for vector in population
    ]

    best_i = int(
        np.argmax(
            [m.fitness for m in metrics]
        )
    )
    best_vector = population[best_i].copy()
    best_metrics = metrics[best_i]

    history: List[dict] = []

    append_history(
        history,
        "de",
        0,
        evaluator,
        encoder,
        bounds,
        best_vector,
        best_metrics,
        metrics,
        population,
    )

    step = 1
    ndim = population.shape[1]

    while evaluator.objective_calls < max_evals:
        completed_any = False

        for i in range(population_size):
            if evaluator.objective_calls >= max_evals:
                break

            choices = [
                j
                for j in range(population_size)
                if j != i
            ]

            r1, r2, r3 = rng.choice(
                choices,
                size=3,
                replace=False,
            )

            mutant = de_mutant(
                population=population,
                r1=int(r1),
                r2=int(r2),
                r3=int(r3),
                differential_weight=differential_weight,
                bounds=bounds,
                nqubits=encoder.nqubits,
                optimize_zz=encoder.optimize_zz,
            )

            trial = population[i].copy()
            j_rand = int(
                rng.integers(
                    0,
                    ndim,
                )
            )

            mask = (
                rng.random(ndim)
                < crossover_probability
            )
            mask[j_rand] = True

            trial[mask] = mutant[mask]

            trial = repair_vector(
                trial,
                bounds,
                encoder.nqubits,
                encoder.optimize_zz,
            )

            trial_metrics = evaluate_vector_safely(
                evaluator,
                trial,
            )

            if (
                trial_metrics.fitness
                >= metrics[i].fitness
            ):
                population[i] = trial
                metrics[i] = trial_metrics

            if (
                trial_metrics.fitness
                > best_metrics.fitness
            ):
                best_vector = trial.copy()
                best_metrics = trial_metrics

            completed_any = True

        append_history(
            history,
            "de",
            step,
            evaluator,
            encoder,
            bounds,
            best_vector,
            best_metrics,
            metrics,
            population,
        )

        step += 1

        if not completed_any:
            break

    return SearchResult(
        optimizer="de",
        best_vector=best_vector,
        best_metrics=best_metrics,
        history=history,
        objective_evaluations=
            int(evaluator.objective_calls),
        unique_candidate_evaluations=
            int(evaluator.unique_candidate_evaluations),
    )


def run_optimizer(
    optimizer: str,
    evaluator: CandidateEvaluator,
    encoder: FluxPulseEncoder,
    bounds: Bounds,
    args: argparse.Namespace,
    seed: int,
) -> SearchResult:

    if optimizer == "ga":
        return search_ga(
            evaluator=evaluator,
            encoder=encoder,
            bounds=bounds,
            population_size=args.population,
            max_evals=args.max_evals,
            elite_fraction=args.elite_fraction,
            mutation_probability=args.mutation_probability,
            mutation_scale=args.mutation_scale,
            seed=seed,
        )

    if optimizer == "random":
        return search_random(
            evaluator=evaluator,
            encoder=encoder,
            bounds=bounds,
            population_size=args.population,
            max_evals=args.max_evals,
            seed=seed,
        )

    if optimizer == "de":
        return search_de(
            evaluator=evaluator,
            encoder=encoder,
            bounds=bounds,
            population_size=args.population,
            max_evals=args.max_evals,
            differential_weight=args.de_f,
            crossover_probability=args.de_cr,
            seed=seed,
        )

    raise ValueError(
        f"unknown optimizer {optimizer!r}"
    )


# =============================================================================
# Final fit/test and feature-map diagnostics
# =============================================================================

def kernel_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict:

    diff = candidate - reference

    ref_norm = float(
        np.linalg.norm(reference)
    )
    diff_norm = float(
        np.linalg.norm(diff)
    )

    return {
        "max_abs_difference":
            float(np.max(np.abs(diff))),
        "frobenius_difference":
            diff_norm,
        "relative_frobenius_difference":
            (
                diff_norm / ref_norm
                if ref_norm > 1.0e-15
                else 0.0
            ),
    }


def replace_tuple_value(
    values: Tuple[float, ...],
    index: int,
    value: float,
) -> Tuple[float, ...]:

    out = list(values)
    out[index] = float(value)
    return tuple(out)


def search_bound_diagnostics(
    vector: np.ndarray,
    bounds: Bounds,
    nqubits: int,
    optimize_zz: bool,
    tolerance: float,
) -> dict:
    matrix = bounds.matrix(nqubits, optimize_zz)
    names = bounds.names(nqubits, optimize_zz)
    values = np.asarray(vector, dtype=np.float64)
    widths = matrix[:, 1] - matrix[:, 0]

    entries: List[dict] = []
    active: List[str] = []

    for name, value, limits, width in zip(
        names,
        values,
        matrix,
        widths,
    ):
        lower_fraction = float((value - limits[0]) / width)
        upper_fraction = float((limits[1] - value) / width)
        nearest_fraction = min(lower_fraction, upper_fraction)
        side = "lower" if lower_fraction <= upper_fraction else "upper"
        is_active = bool(nearest_fraction <= tolerance)
        if is_active:
            active.append(name)

        entries.append(
            {
                "name": name,
                "value": float(value),
                "lower": float(limits[0]),
                "upper": float(limits[1]),
                "nearest_relative_distance": float(nearest_fraction),
                "nearest_side": side,
                "active": is_active,
            }
        )

    return {
        "relative_tolerance": float(tolerance),
        "active_parameters": active,
        "any_active": bool(active),
        "parameters": entries,
    }


def feature_map_diagnostics(
    encoder: FluxPulseEncoder,
    split: DatasetSplit,
    params: EncodingParams,
    K_dev: np.ndarray,
    workdir: Path,
    diagnostic_samples: int,
    eigen_threshold: float,
    best_vector: np.ndarray,
    bounds: Bounds,
    bound_tolerance: float,
) -> dict:

    spectrum = kernel_spectrum(
        K_dev,
        relative_threshold=eigen_threshold,
    )

    n = len(split.X_dev)
    m = min(
        max(4, diagnostic_samples),
        n,
    )

    diagnostic_indices = np.unique(
        np.linspace(
            0,
            n - 1,
            m,
            dtype=np.int64,
        )
    )

    X_probe = split.X_dev[
        diagnostic_indices
    ]

    K_reference = K_dev[
        np.ix_(
            diagnostic_indices,
            diagnostic_indices,
        )
    ]

    delta_sign_probes: List[dict] = []

    for q in range(encoder.nqubits):
        sign_params = EncodingParams(
            alpha=params.alpha,
            delta=replace_tuple_value(
                params.delta,
                q,
                -params.delta[q],
            ),
            duration=params.duration,
            interacting=params.interacting,
        )

        sign_states = encoder.encode_dataset(
            X=X_probe,
            params=sign_params,
            workdir=workdir,
            prefix=f"diag_delta_sign_q{q}",
        )

        K_sign = fidelity_kernel(
            sign_states,
            sign_states,
        )

        delta_sign_probes.append(
            {
                "qubit": int(q),
                "original_delta":
                    float(params.delta[q]),
                "flipped_delta":
                    float(-params.delta[q]),
                **kernel_difference(
                    K_reference,
                    K_sign,
                ),
            }
        )

    diagnostics = {
        "kernel_spectrum": spectrum,
        "diagnostic_sample_count":
            int(len(diagnostic_indices)),
        "detuning_sign_invariance_probes":
            delta_sign_probes,
        "search_bounds": search_bound_diagnostics(
            vector=best_vector,
            bounds=bounds,
            nqubits=encoder.nqubits,
            optimize_zz=encoder.optimize_zz,
            tolerance=bound_tolerance,
        ),
        "fixed_reference_zz_edges": copy.deepcopy(encoder.zz_edges),
        "derived_zz_angles": [
            float(edge["strength"] * params.duration)
            for edge in encoder.zz_edges
        ],
        "product_template_expected_symmetries":
            bool(not encoder.optimize_zz),
        "product_kernel_theoretical_rank_upper_bound_if_ideal_assumptions_hold":
            (
                int(3 ** encoder.nqubits)
                if not encoder.optimize_zz
                else None
            ),
    }

    print(
        "[diagnostic] "
        f"rank={spectrum['numerical_rank']} "
        f"effective_rank={spectrum['effective_rank_entropy']:.4f} "
        f"participation_ratio={spectrum['participation_ratio']:.4f}"
    )

    for probe in delta_sign_probes:
        print(
            "[diagnostic] "
            f"delta sign q{probe['qubit']}: "
            f"max|dK|={probe['max_abs_difference']:.3e} "
            f"relF={probe['relative_frobenius_difference']:.3e}"
        )

    bound_info = diagnostics["search_bounds"]
    if bound_info["any_active"]:
        print(
            "[diagnostic] active search bounds: "
            + ",".join(bound_info["active_parameters"])
        )
    else:
        print("[diagnostic] active search bounds: none")

    return diagnostics


def final_quantum_evaluation(
    encoder: FluxPulseEncoder,
    split: DatasetSplit,
    best_vector: np.ndarray,
    svm_c: float,
    workdir: Path,
    diagnostic_samples: int,
    eigen_threshold: float,
    bounds: Bounds,
    bound_tolerance: float,
    run_diagnostics: bool,
) -> dict:

    params = vector_to_params(
        best_vector,
        encoder.nqubits,
        encoder.optimize_zz,
        encoder.pulse_duration,
    )

    dev_states = encoder.encode_dataset(
        X=split.X_dev,
        params=params,
        workdir=workdir,
        prefix="final_dev",
    )

    test_states = encoder.encode_dataset(
        X=split.X_test,
        params=params,
        workdir=workdir,
        prefix="final_test",
    )

    K_dev = fidelity_kernel(
        dev_states,
        dev_states,
    )
    K_test = fidelity_kernel(
        test_states,
        dev_states,
    )

    clf = SVC(
        kernel="precomputed",
        C=svm_c,
    )
    clf.fit(
        K_dev,
        split.y_dev,
    )

    prediction = clf.predict(
        K_test
    )

    diagnostics = None

    if run_diagnostics:
        diagnostics = feature_map_diagnostics(
            encoder=encoder,
            split=split,
            params=params,
            K_dev=K_dev,
            workdir=workdir,
            diagnostic_samples=diagnostic_samples,
            eigen_threshold=eigen_threshold,
            best_vector=best_vector,
            bounds=bounds,
            bound_tolerance=bound_tolerance,
        )
    else:
        diagnostics = {
            "kernel_spectrum": kernel_spectrum(
                K_dev,
                relative_threshold=eigen_threshold,
            ),
            "search_bounds": search_bound_diagnostics(
                vector=best_vector,
                bounds=bounds,
                nqubits=encoder.nqubits,
                optimize_zz=encoder.optimize_zz,
                tolerance=bound_tolerance,
            ),
        }

    return {
        "test_accuracy": float(
            accuracy_score(
                split.y_test,
                prediction,
            )
        ),
        "dev_alignment": float(
            centered_kernel_alignment(
                K_dev,
                split.y_dev,
            )
        ),
        "params": params,
        "diagnostics": diagnostics,
        "dev_states": dev_states,
        "test_states": test_states,
        "K_dev": K_dev,
        "K_test": K_test,
        "y_dev": split.y_dev,
        "y_test": split.y_test,
        "test_prediction": prediction,
    }


# =============================================================================
# Classical baselines
# =============================================================================

def classical_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    svm_c: float,
    kernel: str,
) -> Tuple[float, float]:

    losses: List[float] = []
    accuracies: List[float] = []

    for train_idx, val_idx in folds:
        clf = SVC(
            kernel=kernel,
            C=svm_c,
            gamma="scale",
        )
        clf.fit(
            X[train_idx],
            y[train_idx],
        )

        decision = np.asarray(
            clf.decision_function(
                X[val_idx]
            ),
            dtype=np.float64,
        ).reshape(-1)

        positive_class = clf.classes_[1]
        y_pm = np.where(
            y[val_idx] == positive_class,
            1.0,
            -1.0,
        )

        losses.append(
            float(
                np.mean(
                    np.maximum(
                        0.0,
                        1.0
                        - y_pm * decision,
                    )
                )
            )
        )

        accuracies.append(
            float(
                accuracy_score(
                    y[val_idx],
                    clf.predict(
                        X[val_idx]
                    ),
                )
            )
        )

    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
    )


def classical_baselines(
    split: DatasetSplit,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    svm_c: float,
) -> dict:

    result = {}

    for kernel in ("linear", "rbf"):
        cv_hinge, cv_accuracy = classical_cv_metrics(
            X=split.X_dev,
            y=split.y_dev,
            folds=folds,
            svm_c=svm_c,
            kernel=kernel,
        )

        clf = SVC(
            kernel=kernel,
            C=svm_c,
            gamma="scale",
        )
        clf.fit(
            split.X_dev,
            split.y_dev,
        )

        test_accuracy = float(
            accuracy_score(
                split.y_test,
                clf.predict(
                    split.X_test
                ),
            )
        )

        result[kernel] = {
            "cv_hinge_loss": cv_hinge,
            "cv_accuracy": cv_accuracy,
            "test_accuracy": test_accuracy,
        }

    return result


# =============================================================================
# Run orchestration
# =============================================================================

def safe_label(path: Path) -> str:
    return "".join(
        c if (
            c.isalnum()
            or c in "_-"
        )
        else "_"
        for c in path.stem
    )


def scalar_stats(
    values: Sequence[float],
) -> dict:

    a = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(a) == 0:
        raise ValueError(
            "cannot calculate statistics for empty sequence"
        )

    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": (
            float(np.std(a, ddof=1))
            if len(a) > 1
            else 0.0
        ),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "values": a.tolist(),
    }


def selected_optimizers(
    optimizer: str,
) -> Tuple[str, ...]:

    if optimizer == "all":
        return OPTIMIZER_ORDER
    return (optimizer,)


def run_one_search(
    args: argparse.Namespace,
    template_path: Path,
    split: DatasetSplit,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    bounds: Bounds,
    optimizer: str,
    seed: int,
    run_dir: Path,
) -> dict:

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with template_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        template = json.load(f)

    configure_reference_zz_strength(
        template,
        (
            args.reference_zz_strength
            if template_optimizes_zz(template)
            else None
        ),
    )

    nqubits = infer_nqubits(
        template
    )
    nfeatures = int(
        split.X_dev.shape[1]
    )

    if nqubits != nfeatures:
        raise ValueError(
            f"{template_path}: {nqubits} qubits but dataset has "
            f"{nfeatures} raw features; one feature per qubit is required"
        )

    encoder = FluxPulseEncoder(
        fluxpulse_binary=Path(
            args.fluxpulse
        ),
        template=template,
        pulse_duration=args.pulse_duration,
        integration_steps=args.integration_steps,
        jobs=args.jobs,
    )

    parameter_names = bounds.names(
        nqubits,
        encoder.optimize_zz,
    )

    mode = (
        "entangling_fixed_hardware_reduced"
        if encoder.optimize_zz
        else "product_reduced"
    )

    print(
        f"[qml] version={SCRIPT_VERSION}"
    )
    print(
        f"[qml] seed={seed} optimizer={optimizer} "
        f"template={template_path} dataset={args.dataset} "
        f"features={nfeatures} nqubits={nqubits} dim={1 << nqubits}"
    )
    print(
        f"[qml] samples: development={len(split.y_dev)} "
        f"test={len(split.y_test)} cv_folds={len(folds)}"
    )
    print(
        f"[qml] parameterization={mode} "
        f"parameters={len(parameter_names)} "
        f"names={','.join(parameter_names)} "
        f"budget={args.max_evals}"
    )
    print(
        "[qml] fitness=-mean_stratified_cv_qsvm_hinge_loss "
        "(test set untouched during search)"
    )
    if encoder.optimize_zz:
        print(
            "[qml] fixed_reference_zz_edges="
            f"{encoder.zz_edges} duration_bounds={list(bounds.duration)}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"fluxpulse_qml_{optimizer}_",
        dir=str(run_dir),
    ) as tmp:

        workdir = Path(tmp)

        evaluator = CandidateEvaluator(
            encoder=encoder,
            X_dev=split.X_dev,
            y_dev=split.y_dev,
            folds=folds,
            svm_c=args.svm_c,
            workdir=workdir,
        )

        search = run_optimizer(
            optimizer=optimizer,
            evaluator=evaluator,
            encoder=encoder,
            bounds=bounds,
            args=args,
            seed=seed,
        )

        final = final_quantum_evaluation(
            encoder=encoder,
            split=split,
            best_vector=search.best_vector,
            svm_c=args.svm_c,
            workdir=workdir,
            diagnostic_samples=args.diagnostic_samples,
            eigen_threshold=args.eigen_threshold,
            bounds=bounds,
            bound_tolerance=args.bound_tolerance,
            run_diagnostics=(
                not args.no_symmetry_diagnostics
            ),
        )

    baselines = classical_baselines(
        split=split,
        folds=folds,
        svm_c=args.svm_c,
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "optimizer": optimizer,
        "template": str(template_path),
        "dataset": args.dataset,
        "seed": int(seed),
        "nqubits": int(nqubits),
        "hilbert_dim": int(1 << nqubits),
        "parameterization": {
            "mode": mode,
            "parameter_count":
                int(len(parameter_names)),
            "parameter_names":
                parameter_names,
            "product_exact_redundancies_removed":
                bool(not encoder.optimize_zz),
            "phase_fixed_to_zero": True,
            "phase_invariance_removed_for_all_one_segment_maps": True,
            "product_detuning_sign_removed":
                bool(not encoder.optimize_zz),
            "fixed_product_pulse_duration": (
                float(args.pulse_duration)
                if not encoder.optimize_zz
                else None
            ),
            "interacting_duration_bounds": (
                list(bounds.duration)
                if encoder.optimize_zz
                else None
            ),
            "alpha_bounds":
                list(bounds.alpha),
            "signed_delta_bounds":
                list(bounds.delta),
            "product_abs_delta_bounds": (
                [
                    0.0,
                    bounds.product_delta_abs_max(),
                ]
                if not encoder.optimize_zz
                else None
            ),
            "fixed_reference_zz_edges":
                copy.deepcopy(encoder.zz_edges),
            "candidate_modifies_hardware_coupling": False,
        },
        "fitness": {
            "name":
                "negative_mean_qsvm_cross_validation_hinge_loss",
            "selection_direction":
                "maximize",
            "cv_folds":
                int(len(folds)),
            "best_fitness":
                float(search.best_metrics.fitness),
            "best_cv_hinge_loss":
                float(search.best_metrics.hinge_loss),
            "best_cv_accuracy":
                float(search.best_metrics.cv_accuracy),
            "best_kernel_alignment_diagnostic":
                float(search.best_metrics.alignment),
            "initial_best_cv_hinge_loss":
                float(search.history[0]["best_cv_hinge_loss"]),
            "absolute_hinge_improvement":
                float(
                    search.history[0]["best_cv_hinge_loss"]
                    - search.best_metrics.hinge_loss
                ),
        },
        "search_budget": {
            "max_candidate_evaluations":
                int(args.max_evals),
            "population":
                int(args.population),
            "boundary_handling":
                "reflection",
            "objective_evaluations":
                int(search.objective_evaluations),
            "unique_candidate_evaluations":
                int(search.unique_candidate_evaluations),
        },
        "best_params":
            final["params"].to_dict(encoder.zz_edges),
        "held_out_test_accuracy":
            float(final["test_accuracy"]),
        "final_development_alignment":
            float(final["dev_alignment"]),
        "feature_map_diagnostics":
            final["diagnostics"],
        "classical_baselines":
            baselines,
        "dataset_metadata":
            split.metadata,
        "history":
            search.history,
    }

    summary_path = (
        run_dir
        / "best_encoding.json"
    )
    data_path = (
        run_dir
        / "best_encoding_data.npz"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    np.savez_compressed(
        data_path,
        dev_states=final["dev_states"],
        test_states=final["test_states"],
        K_dev=final["K_dev"],
        K_test=final["K_test"],
        y_dev=final["y_dev"],
        y_test=final["y_test"],
        test_prediction=
            final["test_prediction"],
    )

    print(
        f"[result] optimizer={optimizer} "
        f"cv_hinge={search.best_metrics.hinge_loss:.6f} "
        f"cv_acc={search.best_metrics.cv_accuracy:.4f} "
        f"align={search.best_metrics.alignment:.6f} "
        f"test_acc={final['test_accuracy']:.4f} "
        f"evals={search.objective_evaluations}"
    )

    print(
        f"[baseline] "
        f"linear cv_acc={baselines['linear']['cv_accuracy']:.4f} "
        f"test={baselines['linear']['test_accuracy']:.4f}; "
        f"rbf cv_acc={baselines['rbf']['cv_accuracy']:.4f} "
        f"test={baselines['rbf']['test_accuracy']:.4f}"
    )

    print(
        f"[qml] wrote {summary_path}"
    )
    print(
        f"[qml] wrote {data_path}"
    )

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "FluxPulse QML optimizer with fixed hardware coupling, reduced "
            "one-segment parameterization, and stratified CV QSVM "
            "hinge-loss fitness."
        )
    )

    parser.add_argument(
        "--version",
        action="version",
        version=SCRIPT_VERSION,
    )

    parser.add_argument(
        "--dataset",
        choices=(
            "moons",
            "circles",
            "spirals",
            "correlated4",
            "breast_cancer_4f",
        ),
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--compare-template",
        default=None,
    )

    parser.add_argument(
        "--fluxpulse",
        default="./build/fluxpulse_run",
    )

    parser.add_argument(
        "--outdir",
        default="out/qml/qsvm_hinge",
    )

    # Dataset
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--moons-noise",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--circles-noise",
        type=float,
        default=0.10,
        help="Gaussian noise added to the concentric-circles coordinates.",
    )

    parser.add_argument(
        "--circles-factor",
        type=float,
        default=0.50,
        help="Inner-to-outer circle radius ratio; must lie in (0,1).",
    )

    parser.add_argument(
        "--spiral-noise",
        type=float,
        default=0.05,
        help="Isotropic Gaussian noise added to the spiral coordinates.",
    )

    parser.add_argument(
        "--spiral-turns",
        type=float,
        default=1.25,
        help="Number of turns made by each spiral arm; must be > 0.",
    )

    parser.add_argument(
        "--correlation",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    # Downstream QSVM CV fitness
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cv-seed-offset",
        type=int,
        default=1000,
        help=(
            "CV split seed is seed + this offset; folds are fixed for all "
            "candidates, optimizers, and paired templates at a given seed."
        ),
    )

    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        choices=(
            "ga",
            "random",
            "de",
            "all",
        ),
        default="de",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-evals",
        type=int,
        default=72,
    )

    # GA
    parser.add_argument(
        "--elite-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.10,
    )

    # DE
    parser.add_argument(
        "--de-f",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--de-cr",
        type=float,
        default=0.90,
    )

    # Pulse variables
    parser.add_argument(
        "--pulse-duration",
        type=float,
        default=0.010,
    )

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--delta-min",
        type=float,
        default=-2.0,
    )

    parser.add_argument(
        "--delta-max",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--duration-min",
        type=float,
        default=0.005,
        help=(
            "Minimum common pulse duration for interacting templates. "
            "Product templates retain --pulse-duration."
        ),
    )

    parser.add_argument(
        "--duration-max",
        type=float,
        default=0.030,
        help=(
            "Maximum common pulse duration for interacting templates."
        ),
    )

    parser.add_argument(
        "--reference-zz-strength",
        type=float,
        default=None,
        help=(
            "Optional run-wide fixed ZZ strength applied to every ZZ edge. "
            "If omitted, each edge keeps the strength in its template. "
            "This value is never optimized."
        ),
    )

    # FluxPulse execution
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
    )

    # Diagnostics
    parser.add_argument(
        "--diagnostic-samples",
        type=int,
        default=24,
        help=(
            "Development samples used for detuning-sign invariance probes."
        ),
    )

    parser.add_argument(
        "--eigen-threshold",
        type=float,
        default=1.0e-8,
        help=(
            "Relative threshold to the largest eigenvalue for numerical rank."
        ),
    )

    parser.add_argument(
        "--bound-tolerance",
        type=float,
        default=1.0e-3,
        help=(
            "Relative distance used to flag an optimized parameter as lying "
            "on a search bound."
        ),
    )

    parser.add_argument(
        "--no-symmetry-diagnostics",
        action="store_true",
        help=(
            "Skip extra FluxPulse runs for detuning-sign probes. "
            "Kernel spectrum is still reported."
        ),
    )

    # Repeated experiment
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--num-seeds",
        "--seeds",
        dest="num_seeds",
        type=int,
        default=1,
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    if args.population < 4:
        raise ValueError(
            "--population must be >= 4"
        )
    if args.max_evals < args.population:
        raise ValueError(
            "--max-evals must be >= --population"
        )
    if args.num_seeds < 1:
        raise ValueError(
            "--num-seeds must be >= 1"
        )
    if args.integration_steps < 4:
        raise ValueError(
            "--integration-steps must be >= 4"
        )
    if args.jobs < 1:
        raise ValueError(
            "--jobs must be >= 1"
        )
    if args.pulse_duration <= 0.0:
        raise ValueError(
            "--pulse-duration must be > 0"
        )
    if not (0.0 < args.test_size < 1.0):
        raise ValueError(
            "--test-size must lie in (0,1)"
        )
    if args.moons_noise < 0.0:
        raise ValueError(
            "--moons-noise must be >= 0"
        )
    if args.circles_noise < 0.0:
        raise ValueError(
            "--circles-noise must be >= 0"
        )
    if not (0.0 < args.circles_factor < 1.0):
        raise ValueError(
            "--circles-factor must lie in (0,1)"
        )
    if not math.isfinite(args.spiral_noise) or args.spiral_noise < 0.0:
        raise ValueError(
            "--spiral-noise must be finite and >= 0"
        )
    if not math.isfinite(args.spiral_turns) or args.spiral_turns <= 0.0:
        raise ValueError(
            "--spiral-turns must be finite and > 0"
        )
    if args.cv_folds < 2:
        raise ValueError(
            "--cv-folds must be >= 2"
        )
    if args.alpha_min < 0.0:
        raise ValueError(
            "--alpha-min must be >= 0 because alpha sign is redundant"
        )
    if args.alpha_min >= args.alpha_max:
        raise ValueError(
            "alpha bounds are invalid"
        )
    if args.delta_min >= args.delta_max:
        raise ValueError(
            "delta bounds are invalid"
        )
    if args.duration_min <= 0.0:
        raise ValueError(
            "--duration-min must be > 0"
        )
    if args.duration_min >= args.duration_max:
        raise ValueError(
            "duration bounds are invalid"
        )
    if args.eigen_threshold <= 0.0:
        raise ValueError(
            "--eigen-threshold must be > 0"
        )
    if not (0.0 <= args.bound_tolerance < 0.5):
        raise ValueError(
            "--bound-tolerance must lie in [0,0.5)"
        )
    if (
        args.reference_zz_strength is not None
        and (
            not math.isfinite(args.reference_zz_strength)
            or abs(args.reference_zz_strength) <= 1.0e-15
        )
    ):
        raise ValueError(
            "--reference-zz-strength must be finite and nonzero"
        )

    primary_template = Path(
        args.template
    )

    compare_template = (
        Path(args.compare_template)
        if args.compare_template
        else None
    )

    if not primary_template.exists():
        raise FileNotFoundError(
            primary_template
        )

    if (
        compare_template is not None
        and not compare_template.exists()
    ):
        raise FileNotFoundError(
            compare_template
        )

    outdir = Path(
        args.outdir
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bounds = Bounds(
        alpha=(
            args.alpha_min,
            args.alpha_max,
        ),
        delta=(
            args.delta_min,
            args.delta_max,
        ),
        duration=(
            args.duration_min,
            args.duration_max,
        ),
    )

    optimizers = selected_optimizers(
        args.optimizer
    )

    seeds = [
        args.seed + i
        for i in range(
            args.num_seeds
        )
    ]

    if args.num_seeds < 5:
        print(
            "[qml] warning: fewer than 5 seeds; aggregate comparisons are "
            "exploratory and should not be used for statistical claims"
        )
    if "random" not in optimizers:
        print(
            "[qml] warning: random search is not enabled; this run cannot "
            "separate optimizer gains from representation gains"
        )

    templates: List[
        Tuple[str, Path]
    ] = [
        (
            "primary",
            primary_template,
        )
    ]

    if compare_template is not None:
        templates.append(
            (
                "comparison",
                compare_template,
            )
        )

    all_results: Dict[
        str,
        Dict[str, List[dict]],
    ] = {
        role: {
            optimizer: []
            for optimizer in optimizers
        }
        for role, _ in templates
    }

    baseline_by_seed: List[dict] = []

    for seed in seeds:
        split = make_dataset_split(
            dataset=args.dataset,
            seed=seed,
            samples=args.samples,
            test_size=args.test_size,
            moons_noise=args.moons_noise,
            circles_noise=args.circles_noise,
            circles_factor=args.circles_factor,
            spiral_noise=args.spiral_noise,
            spiral_turns=args.spiral_turns,
            correlation=args.correlation,
        )

        folds = make_fixed_cv_folds(
            y_dev=split.y_dev,
            n_splits=args.cv_folds,
            seed=(
                seed
                + args.cv_seed_offset
            ),
        )

        baseline_by_seed.append(
            classical_baselines(
                split=split,
                folds=folds,
                svm_c=args.svm_c,
            )
        )

        for role, template_path in templates:
            label = safe_label(
                template_path
            )

            print(
                "\n"
                f"[qml] ===== seed {seed}: {label} ====="
            )

            seed_results: Dict[
                str,
                dict,
            ] = {}

            for optimizer in optimizers:
                print(
                    "\n"
                    f"[qml] --- optimizer {optimizer} ---"
                )

                run_dir = (
                    outdir
                    / f"seed_{seed:04d}"
                    / label
                    / optimizer
                )

                summary = run_one_search(
                    args=args,
                    template_path=template_path,
                    split=split,
                    folds=folds,
                    bounds=bounds,
                    optimizer=optimizer,
                    seed=seed,
                    run_dir=run_dir,
                )

                all_results[
                    role
                ][
                    optimizer
                ].append(
                    summary
                )

                seed_results[
                    optimizer
                ] = summary

            if len(optimizers) > 1:
                print(
                    f"\n[optimizer-compare] seed={seed} template={label}"
                )

                for optimizer in optimizers:
                    s = seed_results[
                        optimizer
                    ]
                    f = s["fitness"]

                    print(
                        f"  {optimizer:6s} "
                        f"hinge={f['best_cv_hinge_loss']:.6f} "
                        f"cv_acc={f['best_cv_accuracy']:.4f} "
                        f"align={f['best_kernel_alignment_diagnostic']:.6f} "
                        f"test={s['held_out_test_accuracy']:.4f}"
                    )

        if compare_template is not None:
            for optimizer in optimizers:
                p = all_results[
                    "primary"
                ][
                    optimizer
                ][
                    -1
                ]

                c = all_results[
                    "comparison"
                ][
                    optimizer
                ][
                    -1
                ]

                delta_hinge = (
                    c["fitness"][
                        "best_cv_hinge_loss"
                    ]
                    - p["fitness"][
                        "best_cv_hinge_loss"
                    ]
                )

                delta_test = (
                    c[
                        "held_out_test_accuracy"
                    ]
                    - p[
                        "held_out_test_accuracy"
                    ]
                )

                print(
                    f"[paired] seed={seed} optimizer={optimizer} "
                    f"delta_hinge={delta_hinge:+.6f} "
                    f"delta_test={delta_test:+.4f}"
                )

    aggregate = {
        "script_version":
            SCRIPT_VERSION,
        "dataset":
            args.dataset,
        "seeds":
            seeds,
        "optimizers":
            list(optimizers),
        "common_candidate_budget":
            int(args.max_evals),
        "boundary_handling":
            "reflection",
        "cv_folds":
            int(args.cv_folds),
        "exploratory_only":
            bool(args.num_seeds < 5),
        "models": {},
        "classical_baselines": {
            "linear_cv_accuracy":
                scalar_stats(
                    [
                        b["linear"]["cv_accuracy"]
                        for b in baseline_by_seed
                    ]
                ),
            "linear_test_accuracy":
                scalar_stats(
                    [
                        b["linear"]["test_accuracy"]
                        for b in baseline_by_seed
                    ]
                ),
            "rbf_cv_accuracy":
                scalar_stats(
                    [
                        b["rbf"]["cv_accuracy"]
                        for b in baseline_by_seed
                    ]
                ),
            "rbf_test_accuracy":
                scalar_stats(
                    [
                        b["rbf"]["test_accuracy"]
                        for b in baseline_by_seed
                    ]
                ),
        },
    }

    for role, template_path in templates:
        label = safe_label(
            template_path
        )

        aggregate[
            "models"
        ][
            role
        ] = {
            "label":
                label,
            "template":
                str(template_path),
            "optimizers": {},
        }

        for optimizer in optimizers:
            runs = all_results[
                role
            ][
                optimizer
            ]

            aggregate[
                "models"
            ][
                role
            ][
                "optimizers"
            ][
                optimizer
            ] = {
                "cv_hinge_loss":
                    scalar_stats(
                        [
                            r["fitness"][
                                "best_cv_hinge_loss"
                            ]
                            for r in runs
                        ]
                    ),
                "cv_accuracy":
                    scalar_stats(
                        [
                            r["fitness"][
                                "best_cv_accuracy"
                            ]
                            for r in runs
                        ]
                    ),
                "kernel_alignment":
                    scalar_stats(
                        [
                            r["fitness"][
                                "best_kernel_alignment_diagnostic"
                            ]
                            for r in runs
                        ]
                    ),
                "test_accuracy":
                    scalar_stats(
                        [
                            r[
                                "held_out_test_accuracy"
                            ]
                            for r in runs
                        ]
                    ),
                "initial_to_final_hinge_improvement":
                    scalar_stats(
                        [
                            r["fitness"]["absolute_hinge_improvement"]
                            for r in runs
                        ]
                    ),
            }

            s = aggregate[
                "models"
            ][
                role
            ][
                "optimizers"
            ][
                optimizer
            ]

            print(
                f"[aggregate] {label}/{optimizer}: "
                f"hinge={s['cv_hinge_loss']['mean']:.6f} "
                f"+/- {s['cv_hinge_loss']['std']:.6f}; "
                f"cv_acc={s['cv_accuracy']['mean']:.4f} "
                f"+/- {s['cv_accuracy']['std']:.4f}; "
                f"test={s['test_accuracy']['mean']:.4f} "
                f"+/- {s['test_accuracy']['std']:.4f}"
            )

        if "random" in optimizers:
            random_runs = all_results[role]["random"]
            optimizer_comparison = {}

            for optimizer in optimizers:
                if optimizer == "random":
                    continue

                candidate_runs = all_results[role][optimizer]
                optimizer_comparison[optimizer] = {
                    "definition": (
                        "positive values favor the named optimizer over "
                        "random search"
                    ),
                    "cv_hinge_improvement": scalar_stats(
                        [
                            random_run["fitness"]["best_cv_hinge_loss"]
                            - candidate_run["fitness"]["best_cv_hinge_loss"]
                            for random_run, candidate_run in zip(
                                random_runs,
                                candidate_runs,
                            )
                        ]
                    ),
                    "cv_accuracy_improvement": scalar_stats(
                        [
                            candidate_run["fitness"]["best_cv_accuracy"]
                            - random_run["fitness"]["best_cv_accuracy"]
                            for random_run, candidate_run in zip(
                                random_runs,
                                candidate_runs,
                            )
                        ]
                    ),
                    "test_accuracy_improvement": scalar_stats(
                        [
                            candidate_run["held_out_test_accuracy"]
                            - random_run["held_out_test_accuracy"]
                            for random_run, candidate_run in zip(
                                random_runs,
                                candidate_runs,
                            )
                        ]
                    ),
                }

            aggregate["models"][role][
                "optimizer_vs_random"
            ] = optimizer_comparison

    if compare_template is not None:
        aggregate[
            "paired_comparison"
        ] = {}

        for optimizer in optimizers:
            primary_runs = all_results[
                "primary"
            ][
                optimizer
            ]
            compare_runs = all_results[
                "comparison"
            ][
                optimizer
            ]

            hinge_differences = [
                c["fitness"][
                    "best_cv_hinge_loss"
                ]
                - p["fitness"][
                    "best_cv_hinge_loss"
                ]
                for p, c in zip(
                    primary_runs,
                    compare_runs,
                )
            ]

            cv_accuracy_differences = [
                c["fitness"]["best_cv_accuracy"]
                - p["fitness"]["best_cv_accuracy"]
                for p, c in zip(
                    primary_runs,
                    compare_runs,
                )
            ]

            test_differences = [
                c[
                    "held_out_test_accuracy"
                ]
                - p[
                    "held_out_test_accuracy"
                ]
                for p, c in zip(
                    primary_runs,
                    compare_runs,
                )
            ]

            aggregate[
                "paired_comparison"
            ][
                optimizer
            ] = {
                "definition":
                    "comparison_minus_primary",
                "cv_hinge_loss_difference":
                    scalar_stats(
                        hinge_differences
                    ),
                "cv_hinge_improvement":
                    scalar_stats(
                        [-value for value in hinge_differences]
                    ),
                "cv_accuracy_difference":
                    scalar_stats(
                        cv_accuracy_differences
                    ),
                "test_accuracy_difference":
                    scalar_stats(
                        test_differences
                    ),
            }

    aggregate_path = (
        outdir
        / "aggregate_summary.json"
    )

    with aggregate_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            aggregate,
            f,
            indent=2,
        )

    print(
        f"[qml] wrote {aggregate_path}"
    )


if __name__ == "__main__":
    main()
