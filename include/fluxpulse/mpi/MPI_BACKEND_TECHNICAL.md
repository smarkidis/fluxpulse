# MPI Backend Technical Notes

## Purpose

This document describes the technical design of the Phase 1 MPI backend for the MCWF ensemble simulator.

The MPI layer is intentionally **above** the solver core. It parallelizes **trajectories**, not the state-vector algebra inside a trajectory. This matches the current MCWF implementation, where each trajectory is evolved independently and ensemble statistics are formed afterward from accumulated first and second moments.

## Design goals

1. Keep `<mpi.h>` out of the core simulation headers.
2. Reuse the existing serial MCWF trajectory path as much as possible.
3. Preserve the existing result semantics:
   - mean expectation values
   - variance
   - standard error of the mean
4. Keep the root rank responsible for final aggregation and file output.
5. Support both:
   - **static scheduling** for reproducibility/debugging
   - **dynamic batching** for throughput/load balancing

## High-level architecture

The backend is split into small components.

### `mpi_context.*`
Thin RAII wrapper around MPI runtime lifetime.

Responsibilities:
- initialize MPI if needed
- finalize MPI if owned by this object
- expose `rank()`, `size()`, `is_root()`
- expose `Barrier()`

It is **not** a general communicator abstraction.

### `mpi_types.hpp`
Contains small POD-like types used by the orchestration layer:
- `SchedulingMode`
- `MpiRunConfig`
- `BatchRequest`
- `BatchAssignment`
- `BatchResultHeader`
- stats views

No MPI symbols appear here.

### `trajectory_batcher.*`
Contains the scheduling logic for trajectory ranges.

Responsibilities:
- compute the static trajectory range for a rank
- generate the next dynamic batch from a global cursor
- normalize batch size
- decide the effective scheduling mode

It knows nothing about MCWF internals.

### `ensemble_accumulator.*`
Owns contiguous local/root accumulation buffers:
- `sum = Σ x`
- `sum_sq = Σ x²`

where `x` is one observable sample from one trajectory.

Responsibilities:
- accumulate one trajectory
- merge partial results from another rank
- finalize into `Result`

### `mpi_runner.*`
Top-level orchestration layer.

Responsibilities:
- execute local trajectory batches
- distribute dynamic work from root to workers
- collect and merge partial statistics
- finalize the root result

## Parallelization model

The backend parallelizes the **ensemble dimension**.

A single MCWF trajectory remains serial on one rank.

This is the right Phase 1 choice because trajectories are independent. The existing serial code already computes ensemble results by repeated calls to the single-trajectory solver and then accumulates statistics. The MPI backend reuses the same idea, but distributes the trajectory ranges across ranks.

## Data layout

The current MCWF accumulation layout is flat and observable-major:

```text
flat_index = obs_index * n_samples + sample_index
```

This is the same layout used by the serial accumulation path.

For each rank, the local accumulator stores:

```text
sum[flat_index]
sum_sq[flat_index]
```

This layout is convenient because:
- it matches the serial code path
- it is contiguous in memory
- it can be sent with one MPI message per buffer

## Result finalization

If `N` trajectories have been accumulated, the final result is computed on root as:

```text
mean   = sum / N
var    = sum_sq / N - mean^2
stderr = sqrt(var / N)
```

The variance here is the same population-style variance already used in the serial application code.

Because finite precision can produce a tiny negative number after `sum_sq / N - mean^2`, the implementation clamps negative values close to zero before taking the square root.

## Sampling rule

The MPI backend must preserve the same time-sampling rule as the serial MCWF path.

Given:
- `n_points = grid.size()`
- `stride = sample_stride`

samples are taken at:
- index `0`
- every `stride`
- and always the last grid point

Equivalent helper logic:

```text
last = n_points - 1
count = last / stride + 1
if last % stride != 0:
    count += 1
```

The root result `tlist` must be built with this exact convention so that serial and MPI outputs remain schema- and index-compatible.

## Static scheduling

Static scheduling assigns one deterministic contiguous trajectory range to each rank.

### Algorithm

Let:
- `n = total_trajectories`
- `p = world_size`
- `r = rank`

Then:

```text
base = n / p
rem  = n % p
count = base + (r < rem ? 1 : 0)
begin = r * base + min(r, rem)
```

Rank `r` receives the half-open range:

```text
[begin, begin + count)
```

### Properties

- deterministic
- reproducible partitioning
- no runtime load balancing
- ideal for debugging and regression tests

### When to use

Use static scheduling when:
- debugging seed behavior
- comparing MPI against serial reference outputs
- running smoke tests
- prioritizing reproducibility over throughput

## Dynamic scheduling

Dynamic scheduling uses **root as scheduler** and worker ranks request more work when they become idle.

### Root-side state

Root maintains:
- `next_traj`: next unassigned trajectory id
- `live_workers`: number of workers not yet terminated
- global accumulator for collected results

### Worker-side state

Each worker repeatedly:
1. sends a work request to root
2. receives a `BatchAssignment`
3. if `terminate == true`, exits
4. otherwise executes the batch locally
5. sends partial statistics back to root
6. requests another batch

### Advantages

- better load balancing
- more robust to per-trajectory runtime variability
- better cluster/node utilization

### Tradeoff

Dynamic scheduling is not intended to be MPI-size invariant by default. Different world sizes can produce different assignment orders and therefore different aggregate floating-point reduction orders.

That is acceptable for the throughput-oriented mode.

## Batch semantics

A batch is a half-open trajectory range:

```text
[traj_begin, traj_begin + traj_count)
```

### Why batches instead of single trajectories?

Batching reduces MPI overhead:
- fewer messages
- fewer root scheduling events
- better amortization of workspace setup

### Batch size tradeoff

Small batches:
- better load balancing
- more MPI traffic

Large batches:
- lower MPI traffic
- potentially worse tail imbalance

A reasonable starting default is a moderate batch size such as `32` or `64`, then tune after performance measurement.

## MPI message protocol

The current protocol uses point-to-point messaging on `MPI_COMM_WORLD`.

### Functions currently used

#### Runtime and world info
- `MPI_Initialized`
- `MPI_Finalized`
- `MPI_Init`
- `MPI_Finalize`
- `MPI_Comm_rank`
- `MPI_Comm_size`
- `MPI_Barrier`

#### Point-to-point communication
- `MPI_Send`
- `MPI_Recv`
- `MPI_Probe`

### Datatype choice

The current implementation sends control structs and numeric buffers using:

```text
MPI_BYTE
```

This is simple and avoids introducing custom MPI datatypes during Phase 1.

Practical implications:
- sender and receiver must agree exactly on struct layout
- this is fine for a homogeneous cluster / same build
- if portability across heterogeneous architectures becomes important later, custom MPI datatypes or explicit field-wise packing would be safer

## Tag usage

The current tag convention separates control messages from data messages.

### Tags

- `kTagRequest = 1001`
- `kTagAssign  = 1002`
- `kTagTrajCount = 1003`
- `kTagSum = 1004`
- `kTagSumSq = 1005`

### Meaning

#### `kTagRequest`
Worker → root.

Payload:
- `BatchRequest`

Purpose:
- indicate that the worker is idle and wants more work

#### `kTagAssign`
Root → worker.

Payload:
- `BatchAssignment`

Purpose:
- assign the next trajectory range
- or terminate the worker when `terminate == true`

#### `kTagTrajCount`
Worker → root.

Payload:
- `TrajectoryId`

Purpose:
- communicate how many trajectories are represented by the following partial sums

#### `kTagSum`
Worker → root.

Payload:
- local `sum` buffer

Purpose:
- communicate `Σ x`

#### `kTagSumSq`
Worker → root.

Payload:
- local `sum_sq` buffer

Purpose:
- communicate `Σ x²`

### Why separate `sum` and `sum_sq` messages?

Simplicity.

This keeps the protocol straightforward and easy to debug. A future optimization could pack both arrays into one contiguous message, but the current split is acceptable for Phase 1.

## Root control loop in dynamic mode

The root loop is driven by incoming messages.

Pseudo-flow:

```text
next_traj = 0
live_workers = world_size - 1
while live_workers > 0:
    probe any source / any tag
    if tag == REQUEST:
        recv request
        assignment = next batch or terminate
        send assignment
        if terminate:
            live_workers -= 1
    else if tag == TRAJCOUNT:
        recv trajectories
        recv sum
        recv sum_sq
        merge into global accumulator
    else:
        error
```

### Why `MPI_Probe`?

Because root must be able to react to two distinct worker events:
- a worker asking for more work
- a worker returning finished data

`MPI_Probe` allows root to inspect the incoming tag first and dispatch correctly.

## Worker loop in dynamic mode

Pseudo-flow:

```text
while true:
    send request
    recv assignment
    if terminate:
        break
    run local batch
    send partial result
```

This keeps workers simple and stateless except for their reusable local workspaces.

## Local batch execution

A local batch is executed by iterating over the assigned trajectory ids.

For each trajectory:
1. derive the trajectory seed from `base_seed` and trajectory id
2. initialize a local RNG
3. run `run_mcwf_trajectory(...)`
4. accumulate the trajectory result into local `sum` and `sum_sq`

### Why use `run_mcwf_trajectory(...)` instead of the serial accumulate helper?

Because the MPI accumulator needs both:
- `Σ x`
- `Σ x²`

The existing accumulate helper adds only `Σ x` directly. Using the full trajectory result gives access to each trajectory’s observable vector, which is then squared locally to update `sum_sq`.

## Reproducibility notes

### What is reproducible

With static scheduling and fixed seeds:
- each rank receives the same deterministic trajectory range for a given `world_size`
- each trajectory id maps deterministically to its seed
- the same run should be repeatable for the same MPI size and build

### What is not guaranteed by default

Dynamic scheduling does **not** guarantee world-size-invariant results bit-for-bit.

Reasons:
- different batch assignment order
- different floating-point merge order
- different timing of worker completion

This is acceptable for throughput mode.

## Why root-only output

Only rank 0 should write:
- result JSON/CSV
- future trace/debug artifacts

Reasons:
- avoids duplicate files
- preserves the existing serial output contract
- keeps reduction/finalization in one place

## Error handling philosophy

The backend currently follows the project’s `Status`-based error handling style.

Typical checks include:
- null output pointers
- invalid dimensions
- buffer size mismatches
- invalid MPI byte counts
- unexpected MPI tags

When an MPI call fails, the current design converts that to a normal project `Status` error rather than immediately aborting the communicator.

This is acceptable for a first backend, though future HPC-oriented versions may want explicit `MPI_Abort` on unrecoverable distributed failures.

## Current limitations

1. No trace/diagnostic support in the MPI app yet.
2. Control messages use raw byte sends instead of custom MPI datatypes.
3. Dynamic mode currently uses root as a pure scheduler/collector rather than also computing local work.
4. No collective reduction path yet.
5. No nonblocking communication yet.
6. No heterogeneous-cluster wire-format guarantees.

## Possible future improvements

### Collectives
Replace the manual gather path in static mode with collective operations such as:
- `MPI_Reduce`
- `MPI_Allreduce`

This could simplify the static path and improve performance.

### Nonblocking messaging
Use:
- `MPI_Isend`
- `MPI_Irecv`
- `MPI_Test` / `MPI_Wait`

This could improve overlap between computation and communication in dynamic mode.

### Packed partial-result messages
Send trajectory count + `sum` + `sum_sq` in one packed payload to reduce message count.

### Root participates in dynamic compute
Allow rank 0 to execute batches while also servicing the scheduler loop, or move scheduling to a dedicated progress strategy.

### Better trace policy
Enable root-only or selected-rank tracing for debug runs.

### Explicit MPI datatypes
Define typed messages for control structs if cross-platform portability becomes important.

## Recommended Phase 1 usage

### Smoke tests / debugging
Use:
- static scheduling
- small MPI size
- fixed seed
- moderate trajectory count

### Throughput experiments
Use:
- dynamic scheduling
- tuned batch size
- root-only result output
- no tracing

## Summary

The Phase 1 MPI backend is a conservative orchestration layer for MCWF ensembles.

It does **not** parallelize a single trajectory. Instead, it distributes trajectory ranges across ranks, accumulates `sum` and `sum_sq` locally, and finalizes the same result statistics already used in the serial code.

The backend currently relies on a simple point-to-point protocol over `MPI_COMM_WORLD`, with explicit tags for work requests, assignments, and partial-result buffers. Static scheduling provides deterministic partitioning, while dynamic batching provides better load balancing.

This keeps the architecture simple, aligned with the serial simulator, and suitable for incremental extension in later phases.
