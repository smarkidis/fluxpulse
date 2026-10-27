# Source layout

This describes the source files present in this distribution.

| Path | Contents |
| --- | --- |
| `CMakeLists.txt` | Core library, serial runner, optional MPI targets and tests |
| `apps/` | Serial and MPI command-line runners |
| `include/fluxpulse/core/`, `math/` | Scalar types, spans, complex arithmetic, RNG, timer and status handling |
| `include/fluxpulse/linalg/`, `src/linalg/` | Dense column-major operators and the GEMV implementation |
| `include/fluxpulse/operators/` | Pauli, ladder and tensor-embedding constructors |
| `include/fluxpulse/control/`, `src/control/` | Time grids, envelope segments, pulses and pulse programs |
| `include/fluxpulse/model/`, `src/model/` | Effective models and collapse operators |
| `include/fluxpulse/sim/`, `src/sim/` | MCWF, Lindblad, observables and results |
| `include/fluxpulse/device/`, `src/device/` | Processor model/program composition and serial ensembles |
| `include/fluxpulse/mpi/`, `src/mpi/` | MPI context, trajectory batching and ensemble aggregation |
| `include/fluxpulse/io/`, `src/io/` | JSON/config readers, compiled-program conversion, result and trace writers |
| `configs/` | Example and experiment inputs |
| `compiled/` | Example compiled instruction inputs |
| `scripts/` | Plotting and research experiments |
| `scaling/` | MPI scaling driver |
| `tests/unit/`, `tests/smoke/` | Standalone assert-based tests |

There is no circuit compiler, scheduler, QASM parser, CUDA backend, or vendor
BLAS backend in this archive. The compiled-program reader converts supported
instructions into pulse segments; it does not supply the missing compiler.

Small types and utilities, including `Pulse`, are header-only. Generated
outputs belong in `out/` and build products in an out-of-source build directory.
