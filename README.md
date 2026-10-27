# FluxPulse

FluxPulse is a C++20 simulator for pulse-driven open quantum dynamics in small
systems. It implements Monte Carlo wave-function (MCWF) trajectories with RK4
integration, a dense Lindblad reference solver, and optional MPI parallelism
over trajectories.

The runners support a two-level qubit, coupled two-level qubits with exchange
and ZZ interactions, and a truncated Duffing transmon. Pulse envelopes are
piecewise constant or piecewise linear. Results are written as JSON or CSV;
the serial runner can also write Chrome/Perfetto trace JSON.

## Build and test

The serial C++ build requires CMake 3.20 or later and a C++20 compiler, with
no third-party library dependencies:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLUXPULSE_ENABLE_MPI=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Tests retain their assertions in Release builds. Use `-DBUILD_TESTING=OFF` to
build only the library and runner.

To build and test the MPI backend, install an MPI implementation and use a
separate build directory:

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DFLUXPULSE_ENABLE_MPI=ON
cmake --build build-mpi --parallel
ctest --test-dir build-mpi --output-on-failure
```

MPI is enabled by default in CMake; set `FLUXPULSE_ENABLE_MPI=OFF` explicitly
for a serial build, as in the first example.

## Run an example

From the repository root:

```bash
mkdir -p out
./build/fluxpulse_run configs/1q_rabi.json
./build/fluxpulse_run configs/1q_ramsey.json
./build/fluxpulse_run configs/1q_t1.json
```

The config controls the output paths; their parent directories must exist.
Use `./build/fluxpulse_run --help` for command-line options.

With MPI enabled:

```bash
mpiexec -n 2 ./build-mpi/fluxpulse_run_mpi configs/1q_t1.json --dynamic --batch_size 32
```

Both runners accept `--compiled_program PATH` to load a compiled instruction
file in place of the config's pulse program. The `compiled/` directory contains
examples. The compiler that produced them is not included in this source tree.

## Conventions and limits

- Subsystem 0 is least significant: basis index `q0 + 2*q1 + 4*q2 + ...`.
- Dense matrices use column-major storage.
- Time units must be consistent with Hamiltonian frequencies and noise times;
  the simulator does not enforce a physical unit system.
- The runners initialize computational basis states. The C++ solver APIs
  accept state-vector or density-matrix inputs.
- A single MCWF trajectory can return a final pure state. An ensemble returns
  observable statistics, without a unique pure-state vector.
- MPI traces are not implemented. Trajectory seeds depend on the global
  trajectory ID; floating-point ensemble reductions can vary with rank count
  and scheduling.
- Use a time step small enough to resolve pulses and keep total MCWF jump
  probability well below one. The integrators have no adaptive step control.
- The config parser also contains oscillator and Kerr-oscillator blocks;
  neither runner implements those models.

## Examples and analysis

`configs/` contains simulation and research experiment configurations.
`scripts/` contains plotting, convergence, leakage, and QML experiments;
these are optional and are not required to build the C++ simulator.
Their imports use NumPy, Matplotlib, scikit-learn, and, for the QML ablation
script, SciPy.

See [runner usage](apps/README.md), [design](docs/DESIGN.md),
[source layout](docs/fluxpulse_tree_structure.md), and
[contribution rules](docs/RULES.md).

## Citation

If you use FluxPulse in scientific work, please cite the accompanying paper:

Stefano Markidis. **FluxPulse: A Pulse-Level Simulator for Superconducting
QPUs.** Q-WORKS 2026, Euro-Par 2026 Workshops.
[Conference program](https://easychair.org/smart-program/Euro-Par2026Workshops/2026-08-24.html).

```bibtex
@inproceedings{markidis2026fluxpulse,
  author    = {Markidis, Stefano},
  title     = {{FluxPulse}: A Pulse-Level Simulator for Superconducting {QPUs}},
  booktitle = {Q-WORKS 2026, Euro-Par 2026 Workshops},
  year      = {2026},
  url       = {https://easychair.org/smart-program/Euro-Par2026Workshops/2026-08-24.html}
}
```

Citation metadata is also provided in [CITATION.cff](CITATION.cff). The reference
uses the confirmed conference information; DOI and proceedings details can be
added when available.

## License

FluxPulse is distributed under the [MIT License](LICENSE).
Citation is appreciated and is not a condition of the license.
