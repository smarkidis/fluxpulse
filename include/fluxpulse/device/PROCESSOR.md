# Processor in FluxPulse (Phase 1)

In FluxPulse, the **Processor** is the high-level façade that orchestrates a simulation run.

It packages together:
- the **effective device model** (Hamiltonians and noise),
- the **pulse program** (time grid + channel envelopes),
- the **solver configuration** (trajectories, seeding, sampling),
and produces a time-series **Result** (expectations vs time).

The goal is to give the rest of the codebase a single, stable entry point for running simulations, while keeping the numerical kernels and physics components modular.

---

## 1) What problem does Processor solve?

FluxPulse has multiple layers:

- **Control layer**: channels, segments, envelopes, pulse program (`u_k(t)`)
- **Model layer**: drift/control operators and collapse operators (`H0`, `Hk`, `Cn`)
- **Solver layer**: RK4 integration, MCWF jumps, measurement (`|ψ(t)⟩` trajectories)
- **I/O layer**: JSON configs, trace export, results writing

Without an orchestrator, every app/test would have to manually:
- seed RNGs per trajectory,
- allocate/reuse solver workspace,
- schedule sampling times,
- average observables across trajectories,
- validate inputs consistently.

**Processor** standardizes this orchestration.

---

## 2) Conceptual interface

A Processor run conceptually looks like:

1) **Validate inputs**
2) **Allocate workspace**
3) For each trajectory:
   - derive deterministic trajectory seed from base seed + trajectory index
   - run one MCWF trajectory
   - accumulate expectation values into sums
4) **Average** sums by number of trajectories
5) Return a **Result** containing:
   - `tlist`
   - observable names (optional)
   - expectation values vs time

In pseudo-notation:

\[
\langle O \rangle(t) \approx \frac{1}{N}\sum_{j=1}^{N}\langle\psi_j(t)|O|\psi_j(t)\rangle
\]

---

## 3) Inputs owned by Processor

### A) ModelBase (device physics)

`Processor.model` is a `ModelBase` with:
- `drift_terms`: time-independent Hamiltonian terms
- `control_terms`: per-channel Hamiltonian terms
- `collapse_ops`: Lindblad operators (noise)

This decouples physics specification from solver code.

### B) PulseProgram (classical controls)

`Processor.program` is a `PulseProgram` with:
- `TimeGrid`: simulation time points (uniform or explicit)
- pulses per channel: envelopes to compute `u_k(t)`

The solver uses this to evaluate control coefficients during evolution.

### C) Observables

`Processor.observables` is a list of operator matrices.
During the run, it computes \(\langle O \rangle\) for each observable at requested times.

Optional `observable_names` provide human-readable labels for output files.

### D) RunOptions

`Processor.opt` contains:
- `trajectories`: number of MCWF trajectories
- `base_seed`: deterministic base seed
- `sample_stride`: how often to record observables (grid decimation)
- `renormalize_each_step`: numerical stability option

---

## 4) Deterministic seeding

MCWF is stochastic but should be reproducible.

Processor uses the convention:
- `seed_traj = derive_trajectory_seed(base_seed, traj_index)`

Each trajectory uses its own independent RNG stream.

This makes results repeatable across runs, and allows tests to assert correctness.

---

## 5) Sampling and Result layout

Processor decides the sampling schedule:

- always includes `t = 0`
- records every `sample_stride` grid points
- always includes the final time point

The output `Result` stores expectation values in a flat array:

- `expectations[obs_index * n_times + time_index]`

This layout is simple to write to JSON/CSV and easy to average.

---

## 6) Phase 1 scope vs Phase 2 direction

### Phase 1
- single-process, single-node
- trajectory loop is serial
- one pulse per channel in PulseProgram (convention)
- MCWF solver is the main execution path

### Phase 2 (typical extensions)
Processor is a natural place to add:
- multithreading or MPI distribution of trajectories
- device “subcomponents” (multiple qubits, couplers) with richer configs
- mixed solver modes (unitary-only, master equation density matrix)
- trace export hooks (Perfetto) for pulse schedules and solver timings

The key design goal is: keep the Processor interface stable while improving internals.

---

## Summary

The **Processor** is the “run orchestrator” of FluxPulse:

- takes `ModelBase` + `PulseProgram` + observables + run options
- runs one or many MCWF trajectories with deterministic seeding
- averages observables into a `Result` time series

It isolates app/test code from solver and RNG details and provides a stable façade for future scaling (Phase 2+).
