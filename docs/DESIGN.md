# FluxPulse design

FluxPulse simulates time-dependent open quantum dynamics with dense operators.
The source includes MCWF trajectories, a Lindblad reference solver, and MPI
parallelism over independent trajectories. It also reads compiled pulse
instructions; a circuit compiler and scheduler are not included.

## Data flow

A config reader constructs a time grid, pulse program, model parameters,
initial-state description, observable names, and run/output options. The
runners build full-system operators and the initial basis state from those
inputs. `Processor` runs serial MCWF ensembles. `MpiRunner` assigns trajectory
ranges to ranks and combines their first and second moments. Result and trace
writers handle output separately from the numerical kernels.

Both runners currently contain some of the same model/state setup code, and
the serial runner has a separate ensemble loop to print progress. These paths
should be consolidated when the runner interfaces are refactored.

## Operators and storage

`ModelBase` owns drift, control, and collapse operators in the full Hilbert
space. A control term associates one operator with one `ChannelId`.
`PulseProgram` associates that channel with a piecewise constant or linear
envelope. Channels are unique within a model and within a program.

`DenseOp` is square and column-major. `lda()` is its leading dimension;
`gemv()` implements `y = alpha*A*x + beta*y` through the GEMV wrapper. The
current wrapper is a CPU loop, without a BLAS dependency. Solver math also
contains some independent dense multiplication helpers.

Owned storage uses vectors. Numerical interfaces use raw pointers or
`Span<T>`. `Complex` is the project's two-scalar complex type. `FLUXPULSE_HD`
annotates small helpers for possible device compilation; it does not make
the current solvers a CUDA implementation.

## Basis and pulse conventions

Subsystem 0 is the least significant digit. For qubits the basis index is
`q0 + 2*q1 + 4*q2 + ...`, corresponding to the ket `|qN ... q1 q0>`.
`kron(A, B)` places `B` on the less significant subsystem.

Envelope segments cover half-open intervals `[t0, t1)`. Finalized envelopes
are sorted and non-overlapping. Gaps evaluate to zero. Solver sampling always
includes the last grid point, including when the stride does not divide the
number of intervals.

## Numerical workspaces

MCWF uses non-Hermitian RK4 propagation and first-order per-step jump
probabilities. The caller must choose a sufficiently small time step. Scratch
vectors, collapse products and envelope cursors are held in `McwfWorkspace`.

The Lindblad solver precomputes the drift sum, collapse adjoints and collapse
products. Its RK4 stage input has a dedicated `rho_stage` buffer: RHS matrix
multiplication scratch must never alias that input. Optional Hermiticity
stabilization updates each conjugate pair together.

## Randomness and ensemble outputs

Each trajectory uses `derive_trajectory_seed(base_seed, global_trajectory_id)`
with a SplitMix64 RNG by default. Individual trajectories retain their seed
when the MPI partition changes. Static and dynamic ensemble aggregation can
produce different floating-point rounding across rank counts or batch orders;
there is no promise of bitwise identical aggregate outputs.

Results contain sampled expectations and, for ensembles, population variance
and the corresponding standard error estimate. Only a single MCWF trajectory
has a final pure-state vector. Diagnostics and traces are supported by the
serial runner; MPI traces are not implemented.

## Validation

Unit and smoke tests are standalone executables using assertions. Assertions
must remain enabled for test targets in every build configuration. Tests cover
basis and matrix conventions, input loading, model construction, analytical
qubit dynamics, and MCWF comparison against the Lindblad solver.
