# Config generation

## Transport chains

[`generate_transport.py`](../scripts/generate_transport.py) generates an open
chain with nearest-neighbor exchange and ZZ interactions for any integer
qubit count >= 2. It starts from the ground state, drives qubit 0, and records
single-excitation populations, ground-state population, and local Z expectations.

```bash
python3 scripts/generate_transport.py --nqubits 6
python3 scripts/generate_transport.py --nqubits 3 --output-dir /tmp/transport
```

See the script's constants and `build_config()` for physical and run settings.
It uses only the Python standard library and writes JSON without running a
simulation. Each invocation writes `configs/transport/Nq_multi.json` for an
ensemble and `Nq_trace.json` for a single trajectory with tracing enabled.
Simulation result paths are set in `build_config()`.

## Experiment sweeps

- `scripts/make_and_run_convergence.py` generates `configs/convergence/`.
- `scripts/make_and_run_duffing_leakage.py` generates `configs/duffing_leakage/`.

Those two scripts also run simulations and replace their generated config
and output directories.
