#!/usr/bin/env python3
"""Post-training interaction ablation for FluxPulse QML feature maps.

For every saved entangling optimum below ``--results-root``, this script:

1. loads the original interaction-on kernels from best_encoding_data.npz;
2. reconstructs the exact development/test split from best_encoding.json;
3. removes the ZZ edges from the FluxPulse template;
4. re-evaluates the same alpha, delta, and pulse duration with ZZ = 0;
5. compares CV hinge loss, CV accuracy, held-out accuracy, kernel spectra,
   state fidelities, and kernel-matrix differences; and
6. writes per-seed and aggregate JSON results.

The pulse parameters are never re-optimized.  Consequently, the comparison
isolates the contribution of simultaneous interaction for the discovered
encoding, rather than comparing two separately trained models.

Typical use from the FluxPulse repository root:

    python3 scripts/qml_interaction_ablation.py \
      --results-root out/qml/paper_spirals_de \
      --qml-script scripts/qml_optimize_encoding.py \
      --fluxpulse ./build/fluxpulse_run \
      --integration-steps 100 \
      --cv-seed-offset 1000 \
      --jobs 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import binomtest, t as student_t
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


SCRIPT_VERSION = "2026-08-12-fixed-pulse-interaction-ablation-v1"


def load_python_module(path: Path) -> Any:
    """Import the QML optimization script without requiring a package."""

    module_name = "qml_optimize_encoding_for_ablation"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import QML module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def scalar_stats(values: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "values": a.tolist(),
    }


def paired_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Two-sided Student-t confidence interval for paired effects."""

    a = np.asarray(values, dtype=np.float64)
    stats = scalar_stats(a)
    if len(a) <= 1:
        low = high = float(a[0])
    else:
        critical = float(
            student_t.ppf(
                0.5 + 0.5 * confidence,
                df=len(a) - 1,
            )
        )
        half_width = critical * float(np.std(a, ddof=1)) / math.sqrt(len(a))
        low = float(np.mean(a) - half_width)
        high = float(np.mean(a) + half_width)
    return {
        **stats,
        "confidence": float(confidence),
        "method": "student_t",
        "lower": low,
        "upper": high,
    }


def resolve_from_project(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def reconstruct_split(qml: Any, summary: dict) -> Any:
    """Rebuild the deterministic split described by best_encoding.json."""

    metadata = summary["dataset_metadata"]
    n_dev = int(metadata["development_samples"])
    n_test = int(metadata["test_samples"])
    n_total = n_dev + n_test
    if n_total <= 0:
        raise ValueError("invalid development/test sample counts")

    return qml.make_dataset_split(
        dataset=str(summary["dataset"]),
        seed=int(summary["seed"]),
        samples=int(metadata.get("samples", n_total)),
        test_size=float(n_test / n_total),
        moons_noise=float(metadata.get("moons_noise", 0.20)),
        circles_noise=float(metadata.get("circles_noise", 0.10)),
        circles_factor=float(metadata.get("circles_factor", 0.50)),
        spiral_noise=float(metadata.get("spiral_noise", 0.05)),
        spiral_turns=float(metadata.get("spiral_turns", 1.25)),
        correlation=float(metadata.get("correlation", 0.60)),
    )


def fit_precomputed_kernel(
    K_dev: np.ndarray,
    K_test: np.ndarray,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    svm_c: float,
) -> Dict[str, Any]:
    clf = SVC(kernel="precomputed", C=svm_c)
    clf.fit(K_dev, y_dev)
    prediction = np.asarray(clf.predict(K_test), dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "prediction": prediction,
        "support_vectors": int(np.sum(clf.n_support_)),
    }


def state_fidelity_summary(
    reference: np.ndarray,
    ablated: np.ndarray,
) -> Dict[str, Any]:
    if reference.shape != ablated.shape:
        raise ValueError(
            f"state-array shape mismatch: {reference.shape} != {ablated.shape}"
        )
    overlaps = np.sum(reference.conj() * ablated, axis=1)
    fidelities = np.abs(overlaps) ** 2
    return scalar_stats(fidelities)


def find_entangling_runs(results_root: Path) -> List[Path]:
    runs: List[Path] = []
    for summary_path in sorted(results_root.rglob("best_encoding.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        edges = (
            summary.get("parameterization", {})
            .get("fixed_reference_zz_edges", [])
        )
        if edges:
            runs.append(summary_path)
    return runs


def ablate_one_run(
    qml: Any,
    summary_path: Path,
    project_root: Path,
    template_override: str | None,
    fluxpulse: Path,
    integration_steps: int,
    jobs: int,
    cv_seed_offset: int,
    svm_c: float,
    eigen_threshold: float,
    overwrite: bool,
) -> dict:
    run_dir = summary_path.parent
    output_path = run_dir / "interaction_ablation.json"
    output_data_path = run_dir / "interaction_ablation_data.npz"

    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[ablation] reusing {output_path}")
        return cached

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    data_path = run_dir / "best_encoding_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"missing reference data: {data_path}")

    reference_data = np.load(data_path)
    required = {
        "dev_states", "test_states", "K_dev", "K_test", "y_dev", "y_test"
    }
    missing = sorted(required.difference(reference_data.files))
    if missing:
        raise ValueError(f"{data_path} is missing arrays: {missing}")

    split = reconstruct_split(qml, summary)
    y_dev_ref = np.asarray(reference_data["y_dev"], dtype=np.int64)
    y_test_ref = np.asarray(reference_data["y_test"], dtype=np.int64)
    if not np.array_equal(split.y_dev, y_dev_ref):
        raise RuntimeError(
            f"{summary_path}: reconstructed development labels differ from "
            "the saved run; check dataset and scikit-learn versions"
        )
    if not np.array_equal(split.y_test, y_test_ref):
        raise RuntimeError(
            f"{summary_path}: reconstructed test labels differ from the saved "
            "run; check dataset and scikit-learn versions"
        )

    cv_folds = int(summary["fitness"]["cv_folds"])
    folds = qml.make_fixed_cv_folds(
        split.y_dev,
        n_splits=cv_folds,
        seed=int(summary["seed"]) + int(cv_seed_offset),
    )
    template_value = template_override or str(summary["template"])
    template_path = resolve_from_project(project_root, template_value)
    with template_path.open("r", encoding="utf-8") as f:
        template_on = json.load(f)

    original_edges = list(template_on.get("model", {}).get("zz", []))
    if not original_edges:
        raise ValueError(f"{template_path} contains no model.zz interaction")

    template_off = json.loads(json.dumps(template_on))
    template_off["model"]["zz"] = []

    best = summary["best_params"]
    dimensionless = best["dimensionless"]
    physical = best["physical"]
    alpha = tuple(float(v) for v in dimensionless["alpha"])
    delta = tuple(float(v) for v in dimensionless["delta"])
    duration = float(physical["pulse_duration"])

    params_off = qml.EncodingParams(
        alpha=alpha,
        delta=delta,
        duration=duration,
        interacting=False,
    )
    encoder_off = qml.FluxPulseEncoder(
        fluxpulse_binary=fluxpulse,
        template=template_off,
        pulse_duration=duration,
        integration_steps=integration_steps,
        jobs=jobs,
    )

    print(
        f"[ablation] seed={summary['seed']} alpha={list(alpha)} "
        f"delta={list(delta)} tau={duration:.8g} ZZ:on->off"
    )

    with tempfile.TemporaryDirectory(
        prefix="fluxpulse_interaction_ablation_",
        dir=str(run_dir),
    ) as tmp:
        workdir = Path(tmp)
        dev_states_off = encoder_off.encode_dataset(
            X=split.X_dev,
            params=params_off,
            workdir=workdir,
            prefix="ablation_zz_off_dev",
        )
        test_states_off = encoder_off.encode_dataset(
            X=split.X_test,
            params=params_off,
            workdir=workdir,
            prefix="ablation_zz_off_test",
        )

    K_dev_on = np.asarray(reference_data["K_dev"], dtype=np.float64)
    K_test_on = np.asarray(reference_data["K_test"], dtype=np.float64)
    dev_states_on = np.asarray(reference_data["dev_states"])
    test_states_on = np.asarray(reference_data["test_states"])

    K_dev_off = qml.fidelity_kernel(dev_states_off, dev_states_off)
    K_test_off = qml.fidelity_kernel(test_states_off, dev_states_off)

    on_cv_hinge, on_cv_accuracy = qml.qsvm_cross_validation_metrics(
        K=K_dev_on,
        y=split.y_dev,
        folds=folds,
        svm_c=svm_c,
    )
    off_cv_hinge, off_cv_accuracy = qml.qsvm_cross_validation_metrics(
        K=K_dev_off,
        y=split.y_dev,
        folds=folds,
        svm_c=svm_c,
    )

    on_test = fit_precomputed_kernel(
        K_dev_on, K_test_on, split.y_dev, split.y_test, svm_c
    )
    off_test = fit_precomputed_kernel(
        K_dev_off, K_test_off, split.y_dev, split.y_test, svm_c
    )

    on_spectrum = qml.kernel_spectrum(K_dev_on, eigen_threshold)
    off_spectrum = qml.kernel_spectrum(K_dev_off, eigen_threshold)

    saved_hinge = float(summary["fitness"]["best_cv_hinge_loss"])
    saved_cv_accuracy = float(summary["fitness"]["best_cv_accuracy"])
    saved_test_accuracy = float(summary["held_out_test_accuracy"])
    consistency_tolerance = 1.0e-10
    consistency_errors = {
        "cv_hinge_loss": float(on_cv_hinge - saved_hinge),
        "cv_accuracy": float(on_cv_accuracy - saved_cv_accuracy),
        "test_accuracy": float(on_test["accuracy"] - saved_test_accuracy),
    }
    if any(
        abs(value) > consistency_tolerance
        for value in consistency_errors.values()
    ):
        raise RuntimeError(
            f"{summary_path}: the saved interaction-on metrics were not "
            f"reproduced (differences={consistency_errors}). Check "
            "--cv-seed-offset, --svm-c, and software versions before "
            "running the ablation."
        )

    result = {
        "script_version": SCRIPT_VERSION,
        "source_summary": str(summary_path),
        "source_script_version": summary.get("script_version"),
        "dataset": summary["dataset"],
        "seed": int(summary["seed"]),
        "ablation": {
            "definition": "same_discovered_pulse_with_reference_zz_removed",
            "parameters_reoptimized": False,
            "alpha": list(alpha),
            "delta": list(delta),
            "pulse_duration": duration,
            "removed_zz_edges": original_edges,
            "integration_steps": int(integration_steps),
            "cv_folds": cv_folds,
            "cv_seed_offset": int(cv_seed_offset),
            "svm_c": svm_c,
        },
        "reference_consistency": {
            "tolerance": consistency_tolerance,
            "reproduced": True,
            "recomputed_minus_saved": consistency_errors,
        },
        "interaction_on": {
            "cv_hinge_loss": float(on_cv_hinge),
            "cv_accuracy": float(on_cv_accuracy),
            "test_accuracy": float(on_test["accuracy"]),
            "development_alignment": float(
                qml.centered_kernel_alignment(K_dev_on, split.y_dev)
            ),
            "support_vectors": int(on_test["support_vectors"]),
            "kernel_spectrum": on_spectrum,
        },
        "interaction_off": {
            "cv_hinge_loss": float(off_cv_hinge),
            "cv_accuracy": float(off_cv_accuracy),
            "test_accuracy": float(off_test["accuracy"]),
            "development_alignment": float(
                qml.centered_kernel_alignment(K_dev_off, split.y_dev)
            ),
            "support_vectors": int(off_test["support_vectors"]),
            "kernel_spectrum": off_spectrum,
        },
        "paired_effect_on_minus_off": {
            "cv_hinge_loss": float(on_cv_hinge - off_cv_hinge),
            "cv_accuracy": float(on_cv_accuracy - off_cv_accuracy),
            "test_accuracy": float(on_test["accuracy"] - off_test["accuracy"]),
            "development_alignment": float(
                qml.centered_kernel_alignment(K_dev_on, split.y_dev)
                - qml.centered_kernel_alignment(K_dev_off, split.y_dev)
            ),
            "effective_rank": float(
                on_spectrum["effective_rank_entropy"]
                - off_spectrum["effective_rank_entropy"]
            ),
        },
        "kernel_difference_off_minus_on": {
            "development": qml.kernel_difference(K_dev_on, K_dev_off),
            "test_to_development": qml.kernel_difference(K_test_on, K_test_off),
        },
        "state_fidelity_on_vs_off": {
            "development": state_fidelity_summary(
                dev_states_on, dev_states_off
            ),
            "test": state_fidelity_summary(test_states_on, test_states_off),
        },
        "prediction_disagreements": int(
            np.sum(on_test["prediction"] != off_test["prediction"])
        ),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    np.savez_compressed(
        output_data_path,
        dev_states_zz_off=dev_states_off,
        test_states_zz_off=test_states_off,
        K_dev_zz_off=K_dev_off,
        K_test_zz_off=K_test_off,
        test_prediction_zz_on=on_test["prediction"],
        test_prediction_zz_off=off_test["prediction"],
        y_dev=split.y_dev,
        y_test=split.y_test,
    )

    print(
        f"[ablation-result] seed={summary['seed']} "
        f"cv_on={on_cv_accuracy:.4f} cv_off={off_cv_accuracy:.4f} "
        f"test_on={on_test['accuracy']:.4f} "
        f"test_off={off_test['accuracy']:.4f} "
        f"delta_test={on_test['accuracy'] - off_test['accuracy']:+.4f}"
    )
    print(f"[ablation] wrote {output_path}")
    print(f"[ablation] wrote {output_data_path}")
    return result


def aggregate_results(results: Sequence[dict]) -> dict:
    ordered = sorted(results, key=lambda item: int(item["seed"]))

    def collect(path: Sequence[str]) -> List[float]:
        values: List[float] = []
        for item in ordered:
            value: Any = item
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    test_gain = collect(("paired_effect_on_minus_off", "test_accuracy"))
    cv_gain = collect(("paired_effect_on_minus_off", "cv_accuracy"))
    hinge_change = collect(("paired_effect_on_minus_off", "cv_hinge_loss"))
    nonzero_test_gain = [value for value in test_gain if value != 0.0]
    positive_nonzero = sum(value > 0.0 for value in nonzero_test_gain)
    sign_test_p = (
        float(
            binomtest(
                positive_nonzero,
                n=len(nonzero_test_gain),
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if nonzero_test_gain
        else 1.0
    )

    return {
        "script_version": SCRIPT_VERSION,
        "definition": "interaction_on_minus_same_pulse_with_zz_off",
        "seeds": [int(item["seed"]) for item in ordered],
        "n": len(ordered),
        "interaction_on": {
            "cv_hinge_loss": scalar_stats(
                collect(("interaction_on", "cv_hinge_loss"))
            ),
            "cv_accuracy": scalar_stats(
                collect(("interaction_on", "cv_accuracy"))
            ),
            "test_accuracy": scalar_stats(
                collect(("interaction_on", "test_accuracy"))
            ),
            "effective_rank": scalar_stats(
                collect((
                    "interaction_on", "kernel_spectrum",
                    "effective_rank_entropy",
                ))
            ),
        },
        "interaction_off": {
            "cv_hinge_loss": scalar_stats(
                collect(("interaction_off", "cv_hinge_loss"))
            ),
            "cv_accuracy": scalar_stats(
                collect(("interaction_off", "cv_accuracy"))
            ),
            "test_accuracy": scalar_stats(
                collect(("interaction_off", "test_accuracy"))
            ),
            "effective_rank": scalar_stats(
                collect((
                    "interaction_off", "kernel_spectrum",
                    "effective_rank_entropy",
                ))
            ),
        },
        "paired_effect_on_minus_off": {
            "cv_hinge_loss": paired_confidence_interval(hinge_change),
            "cv_accuracy": paired_confidence_interval(cv_gain),
            "test_accuracy": paired_confidence_interval(test_gain),
            "effective_rank": paired_confidence_interval(
                collect(("paired_effect_on_minus_off", "effective_rank"))
            ),
            "positive_test_gain_count": int(np.sum(np.asarray(test_gain) > 0.0)),
            "zero_test_gain_count": int(np.sum(np.asarray(test_gain) == 0.0)),
            "negative_test_gain_count": int(np.sum(np.asarray(test_gain) < 0.0)),
            "exact_two_sided_sign_test_p": sign_test_p,
            "sign_test_zero_differences_omitted": int(
                len(test_gain) - len(nonzero_test_gain)
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved entangling pulse encodings with their ZZ "
            "interaction removed and all discovered pulse parameters fixed."
        )
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--qml-script",
        default="scripts/qml_optimize_encoding.py",
        help="Optimization script whose FluxPulse helpers generated the run.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Base directory for relative script, template, and binary paths.",
    )
    parser.add_argument("--template", default=None)
    parser.add_argument("--fluxpulse", default="./build/fluxpulse_run")
    parser.add_argument("--integration-steps", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cv-seed-offset", type=int, default=1000)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--eigen-threshold", type=float, default=1.0e-8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    results_root = resolve_from_project(project_root, args.results_root)
    qml_script = resolve_from_project(project_root, args.qml_script)
    fluxpulse = resolve_from_project(project_root, args.fluxpulse)

    if args.integration_steps < 4:
        raise ValueError("--integration-steps must be >= 4")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if not results_root.is_dir():
        raise FileNotFoundError(f"results root not found: {results_root}")
    if not qml_script.is_file():
        raise FileNotFoundError(f"QML script not found: {qml_script}")
    if not fluxpulse.is_file():
        raise FileNotFoundError(f"FluxPulse executable not found: {fluxpulse}")

    qml = load_python_module(qml_script)
    summary_paths = find_entangling_runs(results_root)
    if not summary_paths:
        raise RuntimeError(
            f"no entangling best_encoding.json files found below {results_root}"
        )

    print(
        f"[ablation] version={SCRIPT_VERSION} runs={len(summary_paths)} "
        f"results_root={results_root}"
    )
    results = [
        ablate_one_run(
            qml=qml,
            summary_path=path,
            project_root=project_root,
            template_override=args.template,
            fluxpulse=fluxpulse,
            integration_steps=args.integration_steps,
            jobs=args.jobs,
            cv_seed_offset=args.cv_seed_offset,
            svm_c=args.svm_c,
            eigen_threshold=args.eigen_threshold,
            overwrite=args.overwrite,
        )
        for path in summary_paths
    ]

    aggregate = aggregate_results(results)
    aggregate_path = results_root / "interaction_ablation_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    effect = aggregate["paired_effect_on_minus_off"]
    print(
        "[ablation-aggregate] "
        f"n={aggregate['n']} "
        f"delta_cv={effect['cv_accuracy']['mean']:+.4f} "
        f"delta_test={effect['test_accuracy']['mean']:+.4f} "
        f"positive/zero/negative="
        f"{effect['positive_test_gain_count']}/"
        f"{effect['zero_test_gain_count']}/"
        f"{effect['negative_test_gain_count']}"
    )
    print(f"[ablation] wrote {aggregate_path}")


if __name__ == "__main__":
    main()
