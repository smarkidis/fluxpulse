#!/usr/bin/env python3
"""Generate ensemble and trace transport-chain configs for N >= 2 qubits.

Only the Python standard library is required; simulations are not run.
"""

import argparse
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DRIVE_AMPLITUDE = 200.0
PULSE_END = 0.0157079633  # Rounded pi/200 duration.
FINAL_TIME = 2.0


def build_config(nqubits=4, trace=False):
    """Build a uniform open chain with an initial X pulse on qubit 0."""
    if not isinstance(nqubits, int) or nqubits < 2:
        raise ValueError("nqubits must be an integer >= 2")

    def edges(strength):
        return [
            {"q0": q, "q1": q + 1, "strength": strength} for q in range(nqubits - 1)
        ]

    pulses = []
    for q in range(nqubits):
        segments = []
        if q == 0:
            segments.append({"t0": 0.0, "t1": PULSE_END, "value": DRIVE_AMPLITUDE})
        segments.append(
            {
                "t0": PULSE_END if q == 0 else 0.0,
                "t1": FINAL_TIME,
                "value": 0.0,
            }
        )
        pulses.append(
            {
                "channel": f"drive_x{q}",
                "envelope": {"type": "piecewise_constant", "segments": segments},
            }
        )

    # Single-excitation projectors, from qubit N-1 down to qubit 0.
    projectors = [
        "P:" + "0" * position + "1" + "0" * (nqubits - position - 1)
        for position in range(nqubits)
    ]
    output_suffix = "" if trace else "_multi"
    return {
        "model": {
            "type": "coupled_qubits",
            "nqubits": nqubits,
            "omega": [0.0] * nqubits,
            "detuning": [0.0] * nqubits,
            "t1": [2.5] * nqubits,
            "tphi": [4.0] * nqubits,
            "enable_drive_x": True,
            "enable_drive_y": False,
            "enable_drive_z": False,
            "exchange_xy": edges(0.9),
            "zz": edges(0.06),
        },
        "initial_state": "ground",
        "program": {
            "grid": {"t0": 0.0, "dt": 0.001, "n_points": 2001},
            "pulses": pulses,
        },
        "run": {
            "trajectories": 1 if trace else 256,
            "base_seed": 12345,
            "sample_stride": 1,
            "renormalize_each_step": True,
        },
        "observables": projectors
        + ["P:" + "0" * nqubits]
        + [f"Z{q}" for q in range(nqubits)],
        "output": {
            "result_path": f"out/{nqubits}q_transport_result{output_suffix}.json",
            "write_trace": trace,
            "trace_path": f"out/{nqubits}q_transport_trace{output_suffix}.json",
        },
        "trace": {
            "emit_segments": True,
            "emit_counters": True,
            "counter_stride": 5,
            "time_scale_to_us": 1000.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nqubits", type=int, default=4, help="qubit count (>= 2)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG_DIR / "transport",
        help="config destination directory (default: configs/transport in this repo)",
    )
    args = parser.parse_args()
    if args.nqubits < 2:
        parser.error("--nqubits must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, trace in (("multi", False), ("trace", True)):
        path = args.output_dir / f"{args.nqubits}q_{variant}.json"
        config = build_config(args.nqubits, trace=trace)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
