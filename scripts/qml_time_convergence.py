#!/usr/bin/env python3
"""
Numerical time-integration convergence check for optimized FluxPulse QML maps.

Re-evaluates an already optimized map at two time-grid resolutions without
rerunning the optimizer.

Default comparison:
    100 integration intervals -> 101 time-grid points
    200 integration intervals -> 201 time-grid points

Reports:
    * relative Frobenius difference between development kernels,
    * relative Frobenius difference between test-to-development kernels,
    * maximum absolute kernel-entry difference,
    * QSVM test accuracy at both resolutions,
    * number of changed test predictions,
    * per-state infidelity 1 - |<psi_coarse|psi_fine>|^2.

The held-out test set is only used for the final downstream comparison.
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
from typing import List, Optional, Tuple

import numpy as np

from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_VERSION = "2026-08-14-time-convergence-v2"

BREAST_CANCER_FEATURES = (
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean concavity",
)


@dataclass(frozen=True)
class EncodingParams:
    alpha: Tuple[float, ...]
    phase: Tuple[float, ...]
    delta: Tuple[float, ...]
    chi: Optional[float]
    fixed_zz_strengths: Optional[Tuple[float, ...]]


@dataclass
class DatasetSplit:
    X_dev: np.ndarray
    X_test: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    for key in ("template", "dataset", "seed", "best_params", "dataset_metadata"):
        if key not in summary:
            raise ValueError(f"{path}: missing required key {key!r}")

    return summary


def params_from_summary(summary: dict) -> EncodingParams:
    best = summary["best_params"]
    if "dimensionless" not in best:
        raise ValueError(
            "best_encoding.json must contain best_params.dimensionless"
        )

    dim = best["dimensionless"]
    alpha = tuple(float(v) for v in dim["alpha"])
    delta = tuple(float(v) for v in dim["delta"])

    phase_raw = dim.get("phase")
    if phase_raw is None:
        phase_fixed = bool(
            summary.get("parameterization", {}).get(
                "phase_fixed_to_zero",
                False,
            )
        )
        if not phase_fixed:
            raise ValueError(
                "optimized phase is missing and the parameterization does "
                "not declare phase_fixed_to_zero"
            )
        phase = tuple(0.0 for _ in alpha)
    else:
        phase = tuple(float(v) for v in phase_raw)

    chi_raw = dim.get("chi")
    chi = None if chi_raw is None else float(chi_raw)

    physical = best.get("physical", {})
    reference_edges = physical.get("reference_zz_edges")
    if reference_edges is None:
        fixed_zz_strengths = None
    else:
        if not isinstance(reference_edges, list):
            raise ValueError(
                "best_params.physical.reference_zz_edges must be a list"
            )
        fixed_zz_strengths = tuple(
            float(edge["strength"])
            for edge in reference_edges
        )

    if not (len(alpha) == len(phase) == len(delta)):
        raise ValueError("inconsistent alpha/phase/delta lengths")

    return EncodingParams(
        alpha=alpha,
        phase=phase,
        delta=delta,
        chi=chi,
        fixed_zz_strengths=fixed_zz_strengths,
    )


def pulse_duration_from_summary(summary: dict) -> float:
    best = summary["best_params"]
    physical = best.get("physical", {})
    if "pulse_duration" in physical:
        tau = float(physical["pulse_duration"])
        if tau <= 0.0:
            raise ValueError("stored pulse_duration must be > 0")
        return tau

    parameterization = summary.get("parameterization", {})
    if "pulse_duration" in parameterization:
        tau = float(parameterization["pulse_duration"])
        if tau <= 0.0:
            raise ValueError("stored pulse_duration must be > 0")
        return tau

    raise ValueError("cannot find pulse duration in best_encoding.json")


def load_raw_dataset_from_metadata(
    dataset: str,
    metadata: dict,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:

    samples = int(metadata.get("samples", 0))

    if dataset == "moons":
        if samples <= 0:
            raise ValueError("moons metadata must contain samples")
        noise = float(metadata.get("moons_noise", 0.20))
        X, y = make_moons(
            n_samples=samples,
            noise=noise,
            random_state=seed,
        )
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)

    if dataset == "spirals":
        if samples < 20:
            raise ValueError("spirals metadata must contain at least 20 samples")

        spiral_noise = float(metadata.get("spiral_noise", 0.0))
        spiral_turns = float(metadata.get("spiral_turns", 1.0))
        radial_floor = float(metadata.get("spiral_radial_floor", 0.03))

        if not math.isfinite(spiral_noise) or spiral_noise < 0.0:
            raise ValueError("spiral_noise must be finite and >= 0")
        if not math.isfinite(spiral_turns) or spiral_turns <= 0.0:
            raise ValueError("spiral_turns must be finite and > 0")
        if not math.isfinite(radial_floor) or not (0.0 <= radial_floor < 1.0):
            raise ValueError("spiral_radial_floor must lie in [0, 1)")

        rng = np.random.default_rng(seed)
        class_sizes = (samples // 2, samples - samples // 2)

        def make_spiral_arm(n_points: int, phase: float) -> np.ndarray:
            radius = np.sqrt(
                rng.uniform(
                    radial_floor,
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

        permutation = rng.permutation(samples)
        return (
            np.asarray(X[permutation], dtype=np.float64),
            np.asarray(y[permutation], dtype=np.int64),
        )

    if dataset == "correlated4":
        if samples <= 0:
            raise ValueError("correlated4 metadata must contain samples")

        rho = float(metadata["correlation"])
        rng = np.random.default_rng(seed)

        z0 = rng.normal(size=samples)
        z1 = rng.normal(size=samples)
        e2 = rng.normal(size=samples)
        e3 = rng.normal(size=samples)

        residual = math.sqrt(max(0.0, 1.0 - rho * rho))

        x0 = z0
        x1 = z1
        x2 = rho * z0 + residual * e2
        x3 = rho * z1 + residual * e3

        X = np.column_stack((x0, x1, x2, x3)).astype(np.float64)
        score = x0 * x1 + x2 * x3
        y = (score > 0.0).astype(np.int64)
        return X, y

    if dataset == "breast_cancer_4f":
        data = load_breast_cancer()
        names = list(data.feature_names)
        indices = [names.index(feature) for feature in BREAST_CANCER_FEATURES]

        X_full = np.asarray(data.data[:, indices], dtype=np.float64)
        y_full = np.asarray(data.target, dtype=np.int64)

        if samples <= 0 or samples >= len(y_full):
            return X_full, y_full

        X, _, y, _ = train_test_split(
            X_full,
            y_full,
            train_size=samples,
            random_state=seed,
            stratify=y_full,
        )
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)

    raise ValueError(f"unsupported dataset {dataset!r}")


def reconstruct_dataset_split(summary: dict) -> DatasetSplit:
    dataset = str(summary["dataset"])
    seed = int(summary["seed"])
    metadata = summary["dataset_metadata"]

    X, y = load_raw_dataset_from_metadata(
        dataset=dataset,
        metadata=metadata,
        seed=seed,
    )

    test_samples = int(metadata.get("test_samples", 0))
    if test_samples <= 0 or test_samples >= len(y):
        raise ValueError(
            "dataset_metadata.test_samples is missing or invalid"
        )

    # Integer test_size reproduces the exact held-out sample count.
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

    def scale(z: np.ndarray) -> np.ndarray:
        return np.clip(z / maxabs, -1.0, 1.0)

    expected_dev = metadata.get("development_samples")
    if expected_dev is not None and int(expected_dev) != len(y_dev):
        raise RuntimeError(
            f"reconstructed development size {len(y_dev)} does not match "
            f"stored value {expected_dev}"
        )

    return DatasetSplit(
        X_dev=scale(X_dev_std),
        X_test=scale(X_test_std),
        y_dev=np.asarray(y_dev, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
    )


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

    raise ValueError(f"cannot infer nqubits from model type {model_type!r}")


def template_zz_edges(template: dict) -> List[dict]:
    zz = template.get("model", {}).get("zz", [])
    if zz is None:
        return []
    if not isinstance(zz, list):
        raise ValueError("model.zz must be a list")
    return zz


def force_ideal_model(cfg: dict, nqubits: int) -> None:
    model = cfg["model"]

    if nqubits == 1 and model.get("type") == "generic_qubit":
        model["t1"] = 0.0
        model["tphi"] = 0.0
    else:
        model["t1"] = [0.0] * nqubits
        model["tphi"] = [0.0] * nqubits

    model["enable_drive_x"] = True
    model["enable_drive_y"] = True


def build_candidate_config(
    template: dict,
    x: np.ndarray,
    params: EncodingParams,
    result_path: Path,
    pulse_duration: float,
    integration_intervals: int,
) -> dict:

    cfg = copy.deepcopy(template)
    nq = infer_nqubits(cfg)

    if len(x) != nq:
        raise ValueError(
            f"sample has {len(x)} features but template has {nq} qubits"
        )
    if integration_intervals < 1:
        raise ValueError("integration_intervals must be >= 1")

    force_ideal_model(cfg, nq)

    detuning = [
        float((params.delta[q] / pulse_duration) * x[q])
        for q in range(nq)
    ]

    model = cfg["model"]
    if nq == 1 and model.get("type") == "generic_qubit":
        model["detuning"] = detuning[0]
    else:
        model["detuning"] = detuning

    zz_edges = template_zz_edges(cfg)

    if zz_edges:
        if params.chi is not None:
            zz_strength = float(params.chi / pulse_duration)
            for edge in zz_edges:
                edge["strength"] = zz_strength
        elif params.fixed_zz_strengths is not None:
            if len(params.fixed_zz_strengths) != len(zz_edges):
                raise ValueError(
                    "stored reference ZZ edge count does not match template"
                )
            for edge, strength in zip(
                zz_edges,
                params.fixed_zz_strengths,
            ):
                edge["strength"] = float(strength)
        else:
            raise ValueError(
                "template contains ZZ edges, but neither optimized chi nor "
                "fixed reference strengths are stored"
            )
    elif params.chi is not None or params.fixed_zz_strengths not in (None, ()):
        raise ValueError(
            "stored ZZ parameters are present but template has no ZZ edges"
        )

    cfg.setdefault("run", {})
    cfg["run"]["trajectories"] = 1
    cfg["run"]["renormalize_each_step"] = True
    cfg["run"]["sample_stride"] = int(integration_intervals)

    cfg.setdefault("output", {})
    cfg["output"]["result_path"] = str(result_path)
    cfg["output"]["write_trace"] = False

    cfg.setdefault("trace", {})
    cfg["trace"]["emit_segments"] = False
    cfg["trace"]["emit_counters"] = False

    # FluxPulse convention:
    # N integration intervals -> N+1 time-grid points.
    dt = float(pulse_duration) / float(integration_intervals)

    cfg.setdefault("program", {})
    cfg["program"]["grid"] = {
        "type": "uniform",
        "t0": 0.0,
        "dt": dt,
        "n_points": int(integration_intervals + 1),
    }

    pulses: List[dict] = []

    for q in range(nq):
        amplitude = float(params.alpha[q] / pulse_duration)

        ux = float(
            amplitude
            * x[q]
            * math.cos(params.phase[q])
        )

        uy = float(
            amplitude
            * x[q]
            * math.sin(params.phase[q])
        )

        pulses.append(
            {
                "channel": f"drive_x{q}",
                "envelope": {
                    "type": "piecewise_constant",
                    "segments": [
                        {
                            "t0": 0.0,
                            "t1": float(pulse_duration),
                            "value": ux,
                        }
                    ],
                },
            }
        )

        pulses.append(
            {
                "channel": f"drive_y{q}",
                "envelope": {
                    "type": "piecewise_constant",
                    "segments": [
                        {
                            "t0": 0.0,
                            "t1": float(pulse_duration),
                            "value": uy,
                        }
                    ],
                },
            }
        )

    cfg["program"]["pulses"] = pulses
    return cfg


def load_final_state(result_path: Path, expected_dim: int) -> np.ndarray:
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    final_state = result.get("final_state")
    if final_state is None:
        raise RuntimeError(f"{result_path}: missing final_state")

    re = np.asarray(final_state.get("real", []), dtype=np.float64)
    im = np.asarray(final_state.get("imag", []), dtype=np.float64)

    if len(re) != expected_dim or len(im) != expected_dim:
        raise RuntimeError(
            f"{result_path}: final-state dimension mismatch; "
            f"got {len(re)}, expected {expected_dim}"
        )

    psi = re + 1j * im
    norm2 = float(np.vdot(psi, psi).real)

    if not np.isfinite(norm2) or norm2 <= 0.0:
        raise RuntimeError(f"{result_path}: invalid norm2={norm2}")

    psi /= math.sqrt(norm2)
    return psi


class FluxPulseEncoder:
    def __init__(
        self,
        fluxpulse_binary: Path,
        template: dict,
        params: EncodingParams,
        pulse_duration: float,
        integration_intervals: int,
        jobs: int,
    ):
        self.binary = fluxpulse_binary.resolve()
        self.template = template
        self.params = params
        self.nqubits = infer_nqubits(template)
        self.dim = 1 << self.nqubits
        self.pulse_duration = float(pulse_duration)
        self.integration_intervals = int(integration_intervals)
        self.jobs = max(1, int(jobs))

        if not self.binary.exists():
            raise FileNotFoundError(
                f"FluxPulse executable not found: {self.binary}"
            )

    def encode_one(
        self,
        x: np.ndarray,
        workdir: Path,
        tag: str,
    ) -> np.ndarray:

        cfg_path = workdir / f"{tag}.json"
        result_path = workdir / f"{tag}_result.json"

        cfg = build_candidate_config(
            template=self.template,
            x=x,
            params=self.params,
            result_path=result_path,
            pulse_duration=self.pulse_duration,
            integration_intervals=self.integration_intervals,
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
                    workdir=workdir,
                    tag=f"{prefix}_{i:05d}",
                )
            return states

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = {
                pool.submit(
                    self.encode_one,
                    X[i],
                    workdir,
                    f"{prefix}_{i:05d}",
                ): i
                for i in range(len(X))
            }

            for future in as_completed(futures):
                i = futures[future]
                states[i] = future.result()

        return states


def fidelity_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    overlap = A.conj() @ B.T
    K = np.abs(overlap) ** 2
    return np.clip(K.real, 0.0, 1.0)


def kernel_difference(coarse: np.ndarray, fine: np.ndarray) -> dict:
    if coarse.shape != fine.shape:
        raise ValueError("kernel shapes differ")

    diff = fine - coarse
    fine_norm = float(np.linalg.norm(fine, ord="fro"))
    coarse_norm = float(np.linalg.norm(coarse, ord="fro"))
    diff_norm = float(np.linalg.norm(diff, ord="fro"))

    reference_norm = fine_norm if fine_norm > 1.0e-15 else coarse_norm
    relative = (
        diff_norm / reference_norm
        if reference_norm > 1.0e-15
        else 0.0
    )

    return {
        "frobenius_difference": diff_norm,
        "relative_frobenius_difference": float(relative),
        "max_absolute_entry_difference": float(
            np.max(np.abs(diff))
        ),
        "mean_absolute_entry_difference": float(
            np.mean(np.abs(diff))
        ),
    }


def state_infidelity_stats(
    coarse: np.ndarray,
    fine: np.ndarray,
) -> dict:

    if coarse.shape != fine.shape:
        raise ValueError("state-array shapes differ")

    overlaps = np.sum(
        coarse.conj() * fine,
        axis=1,
    )

    fidelities = np.clip(
        (np.abs(overlaps) ** 2).real,
        0.0,
        1.0,
    )

    infidelity = 1.0 - fidelities

    return {
        "n_states": int(len(infidelity)),
        "mean_infidelity": float(np.mean(infidelity)),
        "median_infidelity": float(np.median(infidelity)),
        "max_infidelity": float(np.max(infidelity)),
        "p95_infidelity": float(np.quantile(infidelity, 0.95)),
        "min_state_fidelity": float(np.min(fidelities)),
    }


def qsvm_test_accuracy(
    K_dev: np.ndarray,
    K_test: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    svm_c: float,
) -> Tuple[float, np.ndarray]:

    clf = SVC(
        kernel="precomputed",
        C=svm_c,
    )
    clf.fit(K_dev, y_dev)
    prediction = clf.predict(K_test)

    return (
        float(accuracy_score(y_test, prediction)),
        prediction,
    )


def safe_label(path: Path) -> str:
    return "".join(
        c if (c.isalnum() or c in "_-") else "_"
        for c in path.stem
    )


def resolve_template_path(
    summary: dict,
    override: Optional[Path],
) -> Path:

    path = override if override is not None else Path(summary["template"])

    if path.exists():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(
        f"cannot find FluxPulse template {path}; "
        "run from the repository root or use --template"
    )


def run_resolution(
    fluxpulse_binary: Path,
    template: dict,
    params: EncodingParams,
    pulse_duration: float,
    intervals: int,
    split: DatasetSplit,
    jobs: int,
    workdir: Path,
    tag: str,
) -> dict:

    encoder = FluxPulseEncoder(
        fluxpulse_binary=fluxpulse_binary,
        template=template,
        params=params,
        pulse_duration=pulse_duration,
        integration_intervals=intervals,
        jobs=jobs,
    )

    print(
        f"[convergence] {tag}: "
        f"intervals={intervals} "
        f"n_points={intervals + 1} "
        f"dt={pulse_duration / intervals:.12g}"
    )

    dev_states = encoder.encode_dataset(
        X=split.X_dev,
        workdir=workdir,
        prefix=f"{tag}_dev",
    )

    test_states = encoder.encode_dataset(
        X=split.X_test,
        workdir=workdir,
        prefix=f"{tag}_test",
    )

    K_dev = fidelity_kernel(dev_states, dev_states)
    K_test = fidelity_kernel(test_states, dev_states)

    return {
        "intervals": int(intervals),
        "n_points": int(intervals + 1),
        "dt": float(pulse_duration / intervals),
        "dev_states": dev_states,
        "test_states": test_states,
        "K_dev": K_dev,
        "K_test": K_test,
    }


def check_one_map(
    best_path: Path,
    fluxpulse_binary: Path,
    coarse_intervals: int,
    fine_intervals: int,
    svm_c: float,
    jobs: int,
    outdir: Path,
    template_override: Optional[Path],
) -> dict:

    summary = load_summary(best_path)
    params = params_from_summary(summary)
    tau = pulse_duration_from_summary(summary)
    split = reconstruct_dataset_split(summary)

    template_path = resolve_template_path(
        summary=summary,
        override=template_override,
    )

    with template_path.open("r", encoding="utf-8") as f:
        template = json.load(f)

    nq = infer_nqubits(template)

    if len(params.alpha) != nq:
        raise ValueError(
            f"{best_path}: optimized parameters imply "
            f"{len(params.alpha)} qubits, template has {nq}"
        )

    if split.X_dev.shape[1] != nq:
        raise ValueError(
            f"{best_path}: dataset has {split.X_dev.shape[1]} features, "
            f"template has {nq} qubits"
        )

    label = (
        safe_label(Path(str(summary["dataset"])))
        + "_"
        + safe_label(template_path)
        + "_"
        + str(summary.get("optimizer", "optimized"))
        + f"_seed{int(summary['seed']):04d}"
    )

    run_outdir = outdir / label
    run_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[convergence] ===== {label} =====")
    print(f"[convergence] source={best_path}")
    print(
        f"[convergence] tau={tau:.12g} "
        f"development={len(split.y_dev)} "
        f"test={len(split.y_test)}"
    )

    with tempfile.TemporaryDirectory(
        prefix="fluxpulse_convergence_",
        dir=str(run_outdir),
    ) as tmp:

        workdir = Path(tmp)

        coarse = run_resolution(
            fluxpulse_binary=fluxpulse_binary,
            template=template,
            params=params,
            pulse_duration=tau,
            intervals=coarse_intervals,
            split=split,
            jobs=jobs,
            workdir=workdir,
            tag="coarse",
        )

        fine = run_resolution(
            fluxpulse_binary=fluxpulse_binary,
            template=template,
            params=params,
            pulse_duration=tau,
            intervals=fine_intervals,
            split=split,
            jobs=jobs,
            workdir=workdir,
            tag="fine",
        )

    dev_kernel_diff = kernel_difference(
        coarse["K_dev"],
        fine["K_dev"],
    )

    test_kernel_diff = kernel_difference(
        coarse["K_test"],
        fine["K_test"],
    )

    dev_infidelity = state_infidelity_stats(
        coarse["dev_states"],
        fine["dev_states"],
    )

    test_infidelity = state_infidelity_stats(
        coarse["test_states"],
        fine["test_states"],
    )

    coarse_accuracy, coarse_pred = qsvm_test_accuracy(
        K_dev=coarse["K_dev"],
        K_test=coarse["K_test"],
        y_dev=split.y_dev,
        y_test=split.y_test,
        svm_c=svm_c,
    )

    fine_accuracy, fine_pred = qsvm_test_accuracy(
        K_dev=fine["K_dev"],
        K_test=fine["K_test"],
        y_dev=split.y_dev,
        y_test=split.y_test,
        svm_c=svm_c,
    )

    prediction_changes = int(
        np.sum(coarse_pred != fine_pred)
    )

    accuracy_change = float(
        fine_accuracy - coarse_accuracy
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "source_best_encoding": str(best_path),
        "template": str(template_path),
        "dataset": summary["dataset"],
        "seed": int(summary["seed"]),
        "optimizer": summary.get("optimizer"),
        "pulse_duration": float(tau),
        "svm_c": float(svm_c),
        "coarse": {
            "integration_intervals": int(coarse_intervals),
            "time_grid_points": int(coarse_intervals + 1),
            "dt": float(tau / coarse_intervals),
            "test_accuracy": coarse_accuracy,
        },
        "fine": {
            "integration_intervals": int(fine_intervals),
            "time_grid_points": int(fine_intervals + 1),
            "dt": float(tau / fine_intervals),
            "test_accuracy": fine_accuracy,
        },
        "kernel_difference": {
            "development_kernel": dev_kernel_diff,
            "test_to_development_kernel": test_kernel_diff,
        },
        "qsvm": {
            "coarse_test_accuracy": coarse_accuracy,
            "fine_test_accuracy": fine_accuracy,
            "fine_minus_coarse_accuracy": accuracy_change,
            "changed_test_predictions": prediction_changes,
            "test_samples": int(len(split.y_test)),
        },
        "state_infidelity": {
            "development": dev_infidelity,
            "test": test_infidelity,
        },
    }

    result_path = run_outdir / "time_convergence.json"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    np.savez_compressed(
        run_outdir / "time_convergence_data.npz",
        K_dev_coarse=coarse["K_dev"],
        K_dev_fine=fine["K_dev"],
        K_test_coarse=coarse["K_test"],
        K_test_fine=fine["K_test"],
        dev_states_coarse=coarse["dev_states"],
        dev_states_fine=fine["dev_states"],
        test_states_coarse=coarse["test_states"],
        test_states_fine=fine["test_states"],
        y_dev=split.y_dev,
        y_test=split.y_test,
        prediction_coarse=coarse_pred,
        prediction_fine=fine_pred,
    )

    print(
        "[convergence] development kernel: "
        f"relative Frobenius difference="
        f"{dev_kernel_diff['relative_frobenius_difference']:.6e}, "
        f"max|dK|="
        f"{dev_kernel_diff['max_absolute_entry_difference']:.6e}"
    )

    print(
        "[convergence] test kernel: "
        f"relative Frobenius difference="
        f"{test_kernel_diff['relative_frobenius_difference']:.6e}, "
        f"max|dK|="
        f"{test_kernel_diff['max_absolute_entry_difference']:.6e}"
    )

    print(
        "[convergence] QSVM: "
        f"accuracy coarse={coarse_accuracy:.6f}, "
        f"fine={fine_accuracy:.6f}, "
        f"delta={accuracy_change:+.6f}, "
        f"changed_predictions={prediction_changes}"
    )

    print(
        "[convergence] state infidelity: "
        f"dev mean={dev_infidelity['mean_infidelity']:.6e}, "
        f"dev max={dev_infidelity['max_infidelity']:.6e}, "
        f"test mean={test_infidelity['mean_infidelity']:.6e}, "
        f"test max={test_infidelity['max_infidelity']:.6e}"
    )

    print(f"[convergence] wrote {result_path}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate optimized FluxPulse QML maps at two "
            "time-integration resolutions."
        )
    )

    parser.add_argument(
        "--version",
        action="version",
        version=SCRIPT_VERSION,
    )

    parser.add_argument(
        "--best",
        nargs="+",
        required=True,
        help=(
            "One or more best_encoding.json files produced by the "
            "QML optimizer."
        ),
    )

    parser.add_argument(
        "--fluxpulse",
        default="./build/fluxpulse_run",
    )

    parser.add_argument(
        "--coarse-intervals",
        type=int,
        default=100,
        help=(
            "Coarse integration intervals. "
            "100 intervals means 101 time-grid points."
        ),
    )

    parser.add_argument(
        "--fine-intervals",
        type=int,
        default=200,
        help=(
            "Fine integration intervals. "
            "200 intervals means 201 time-grid points."
        ),
    )

    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--outdir",
        default="out/qml/time_convergence",
    )

    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Optional template override; valid only with one --best file."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.coarse_intervals < 1:
        raise ValueError("--coarse-intervals must be >= 1")

    if args.fine_intervals <= args.coarse_intervals:
        raise ValueError(
            "--fine-intervals must be greater than --coarse-intervals"
        )

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    if args.svm_c <= 0.0:
        raise ValueError("--svm-c must be > 0")

    best_paths = [Path(path) for path in args.best]

    for path in best_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    template_override = (
        Path(args.template)
        if args.template
        else None
    )

    if template_override is not None and len(best_paths) != 1:
        raise ValueError(
            "--template override can only be used with one --best file"
        )

    if template_override is not None and not template_override.exists():
        raise FileNotFoundError(template_override)

    fluxpulse_binary = Path(args.fluxpulse)
    if not fluxpulse_binary.exists():
        raise FileNotFoundError(fluxpulse_binary)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []

    for best_path in best_paths:
        results.append(
            check_one_map(
                best_path=best_path,
                fluxpulse_binary=fluxpulse_binary,
                coarse_intervals=args.coarse_intervals,
                fine_intervals=args.fine_intervals,
                svm_c=args.svm_c,
                jobs=args.jobs,
                outdir=outdir,
                template_override=template_override,
            )
        )

    aggregate = {
        "script_version": SCRIPT_VERSION,
        "coarse_intervals": int(args.coarse_intervals),
        "coarse_time_grid_points": int(args.coarse_intervals + 1),
        "fine_intervals": int(args.fine_intervals),
        "fine_time_grid_points": int(args.fine_intervals + 1),
        "runs": results,
    }

    aggregate_path = outdir / "time_convergence_summary.json"

    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(
        "\n"
        f"[convergence] wrote aggregate summary "
        f"{aggregate_path}"
    )


if __name__ == "__main__":
    main()
