# MPI Backend

## Purpose

This document describes the Phase 1 MPI backend for FluxPulse MCWF ensemble runs.

The backend parallelizes **independent MCWF trajectories** across MPI ranks while keeping the
core solver, model, pulse program, and observable code free of MPI dependencies.

The design goal is to preserve the existing serial execution model:

- one trajectory uses the existing MCWF solver
- many trajectories are combined through first and second moments
- the final result contains mean expectation values, variance, and standard error

MPI is therefore an **orchestration layer above the serial solver**, not a rewrite of the solver.

---

## Scope

The MPI backend currently targets:

- MCWF ensemble execution
- root-only aggregation and output writing
- optional static scheduling for reproducibility
- dynamic batching for load balancing

It does **not** change:

- model construction
- pulse-program semantics
- observable definitions
- per-trajectory MCWF stepping
- result JSON/CSV schema

---

## Design principles

### 1. Keep MPI out of core simulation headers

MPI support should remain in `include/fluxpulse/mpi/` and `src/mpi/`.

Core simulation headers such as model, solver, observables, result, and processor logic should not
include `<mpi.h>`.

This keeps the codebase easier to test, easier to compile without MPI, and easier to extend later.

### 2. Parallelize trajectories, not time steps

MCWF trajectories are independent once the model, pulse program, observables, initial state, and
base seed are fixed.

That makes trajectory-level parallelism the natural MPI decomposition.

Each rank computes one or more trajectory batches and returns partial statistics.

### 3. Match the serial accumulation semantics exactly

The serial multi-trajectory path already computes:

- `sum = Σ x`
- `sum_sq = Σ x^2`

and then finalizes:

- `mean = sum / N`
- `var = sum_sq / N - mean^2`
- `stderr = sqrt(var / N)`

The MPI backend must preserve this exact meaning so that serial and MPI results agree up to normal
floating-point and Monte Carlo differences.

### 4. Root rank owns final output

Only rank 0 should:

- finalize the global `Result`
- attach sampled `tlist`
- attach observable names
- write JSON / CSV result files
- print final run summaries

Non-root ranks should only participate in computation and communication.

### 5. Reuse the existing JSON configuration flow

The MPI app should use the same config files and the same model / observable / initial-state build
path as the serial app.

MPI-specific behavior is introduced first through CLI options, before extending the config reader.

---

## High-level architecture

### Serial reference path

The serial MCWF ensemble path works as follows:

1. Build `Processor` from config-derived model, program, observables, and run options.
2. For each trajectory:
   - derive a deterministic trajectory seed from `base_seed` and `traj_id`
   - run one MCWF trajectory
   - accumulate `sum` and `sum_sq`
3. Finalize the `Result` using mean / variance / stderr.
4. Fill sampled times consistently with the solver sampling rule.

The MPI path mirrors this, but distributes trajectory batches across ranks.

### MPI path

1. All ranks build the same immutable simulation inputs locally:
   - `Processor`
   - initial state `psi0`
2. Work is partitioned into trajectory batches.
3. Each worker rank runs local batches using the existing serial trajectory solver.
4. Each rank produces local partial statistics:
   - local trajectory count
   - local `sum`
   - local `sum_sq`
5. Rank 0 merges all partial statistics.
6. Rank 0 finalizes the global `Result` and writes output.

---

## Main components

### `mpi_types.hpp`

Small MPI-facing POD-style types used by the orchestration layer.

Typical contents:

- scheduling mode enum
- MPI run config
- batch request / assignment structs
- local ensemble stats views

These types should not depend on `<mpi.h>`.

### `mpi_context.hpp/.cpp`

A minimal RAII wrapper for MPI runtime lifetime and world metadata.

Responsibilities:

- initialize MPI if needed
- finalize MPI if owned by the wrapper
- expose `rank()`, `size()`, `is_root()`
- expose `Barrier()`

This is **not** intended to be a general communicator abstraction.

### `trajectory_batcher.hpp/.cpp`

Defines how trajectory work is split.

Responsibilities:

- static partitioning for reproducible mode
- dynamic batch generation for load-balanced mode
- batch-size normalization helpers

This layer knows only about trajectory ids and counts, not about solver internals.

### `ensemble_accumulator.hpp/.cpp`

Owns local or root-side contiguous statistics buffers.

Responsibilities:

- store `sum` and `sum_sq`
- accumulate one trajectory result
- merge another accumulator or stats view
- finalize into `Result`

The storage layout matches the serial solver convention:

`flat_index = obs_index * n_samples + sample_index`

### `mpi_runner.hpp/.cpp`

Top-level MPI orchestration layer.

Responsibilities:

- infer the sampled result shape from `Processor`
- run one assigned batch locally
- execute static or dynamic MPI scheduling
- gather and merge partial statistics
- finalize the root result

This is the main coordination point for the MPI backend.

### `apps/fluxpulse_run_mpi.cpp`

MPI front-end application.

Responsibilities:

- parse CLI arguments
- load the same JSON config as the serial runner
- optionally override the pulse program via `--compiled_program`
- build `Processor` and initial state
- launch the MPI run
- write root-only output

---

## Scheduling modes

### Static scheduling

Static scheduling partitions the full trajectory range once across all ranks.

Properties:

- deterministic partitioning
- simple communication pattern
- good for reproducibility and debugging
- less flexible under load imbalance

Typical use:

- smoke tests
- deterministic comparisons with serial runs
- debugging seed or reduction issues

### Dynamic scheduling

Dynamic scheduling uses a server / worker pattern.

Properties:

- rank 0 distributes batches on demand
- workers request more work when idle
- better load balancing when trajectory cost varies
- communication is more involved

Typical use:

- larger production ensemble runs
- heterogeneous machines or noisy workload distributions

### Reproducible mode

Reproducible mode should force static scheduling.

The goal is not bitwise equality across every possible MPI size, but a deterministic and debuggable
execution policy for a fixed configuration.

---

## Data flow

### Local batch execution

For a single assigned batch:

1. allocate or reuse one `McwfWorkspace`
2. allocate or reuse one per-trajectory `Result`
3. for each trajectory id in the assigned half-open range:
   - derive trajectory seed
   - run `mcwf::run_mcwf_trajectory(...)`
   - accumulate `sum` and `sum_sq`
4. return local partial statistics

This deliberately reuses the existing serial per-trajectory solver instead of duplicating solver logic.

### Root-side reduction

Rank 0 receives partial results from all other ranks and merges them.

Once all trajectories are accounted for, rank 0 finalizes:

- expectation means
- variances
- standard errors
- sampled time list
- observable names

---

## Result semantics

The MPI backend must preserve the serial result semantics.

### Sampling rule

The sampled time list follows the same convention as the serial app and processor:

- include index `0`
- include every `sample_stride` point
- always include the last grid point even if it is not aligned with the stride

### Observable layout

Flattened observable storage is ordered as:

- all sampled times for observable 0
- all sampled times for observable 1
- and so on

Equivalent flat index:

`obs_index * n_samples + sample_index`

### Uncertainty fields

For `N` trajectories:

- `mean = sum / N`
- `var = sum_sq / N - mean^2`
- clamp very small negative `var` to zero for numerical safety
- `stderr = sqrt(var / N)`

These match the serial multi-trajectory implementation.

---

## Determinism and seeds

Each trajectory must be assigned a deterministic seed based on:

- global base seed
- trajectory id

That means MPI should preserve the same trajectory-to-seed rule as the serial implementation.

This is essential for:

- reproducibility
- comparing serial and MPI results
- debugging scheduling changes without changing trajectory identities

The exact order of floating-point reductions can still differ between modes, especially in dynamic
batching, so strict bitwise identity is not the default expectation.

---

## Trace and diagnostics policy

For Phase 1 MPI support, trace generation should stay conservative.

Recommended policy:

- result output: enabled on root only
- trace output: disabled in the MPI app initially
- diagnostics: only meaningful for single-trajectory or debug-oriented runs

The serial app already treats detailed diagnostics and trace counters as most useful when
`trajectories == 1`.

Later, trace support can be extended with a rank-aware policy, for example root-only or a dedicated
trace rank.

---

## Current CLI approach

Before the config reader is extended, the MPI app can reuse the existing JSON files and add MPI
control through CLI flags such as:

- `--batch_size <N>`
- `--static`
- `--dynamic`
- `--compiled_program <path>`

This keeps Week 5 focused on backend architecture instead of broad schema changes.

---

## Validation strategy

### Unit validation

- `test_ensemble_accumulator.cpp`
  - accumulation
  - merge
  - finalize into `Result`

### MPI smoke tests

- `smoketest_mpi_1q_t1.cpp`
  - compare decay anchors against analytic expectation
- `smoketest_mpi_2q_swap.cpp`
  - compare coherent swap anchors against known oscillation behavior

### Serial / MPI consistency checks

For fixed seed and static scheduling:

- compare serial and MPI outputs on the same config
- check `tlist`, observable names, and uncertainty fields
- allow small floating-point differences where reduction order changes

---

## Current limitations

At this stage, the MPI backend is intentionally limited:

- no custom MPI config schema yet
- no full trace export path in the MPI app yet
- no subcommunicator support
- no hybrid MPI + thread parallelism policy yet
- no rank-local checkpointing or restart support

These are acceptable for Phase 1, whose goal is a working and maintainable MPI ensemble backend.

---

## Future extensions

Natural next steps after the initial MPI backend are:

- add MPI fields to config loading
- support explicit reproducible mode in JSON
- add root-side progress reporting for dynamic mode
- add optional root-side reduction via collective MPI calls where appropriate
- add rank-aware trace or debug output
- evaluate performance tradeoffs between per-trajectory results and direct local accumulation paths

---

## Summary

The FluxPulse MPI backend is a thin orchestration layer for MCWF ensembles.

Its core ideas are:

- parallelize independent trajectories across ranks
- keep MPI out of the solver core
- preserve the serial result semantics exactly
- centralize final aggregation and output on rank 0
- support both static reproducible scheduling and dynamic load-balanced batching

This design keeps the backend aligned with the current serial implementation while leaving room for
future performance and usability improvements.
