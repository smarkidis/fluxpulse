# Lindblad Master-Equation Solver (Phase 1 Validation)

This project now includes a small **density-matrix (Lindblad) reference solver** intended to validate the MCWF
implementation on small Hilbert spaces (typically 1–2 qubits, optionally 3).

Files:
- `include/fluxpulse/sim/lindblad/lindblad_solver.hpp`
- `src/sim/lindblad/lindblad_solver.cpp`

Design choice: **Option A (solver-only)** — the Lindblad right-hand side is implemented as a `static` helper inside
`lindblad_solver.cpp` (no separate `lindblad_rhs.*` yet). This keeps Phase 1 simple and minimizes API surface.

---

## 1. Equation solved

The solver integrates the Lindblad master equation:

\[
\dot\rho(t) = -i[H(t),\rho(t)] + \sum_j \left(
C_j\rho(t)C_j^\dagger - \tfrac{1}{2}\{C_j^\dagger C_j,\rho(t)\}
\right).
\]

where:
- \(\rho\) is the density matrix (Hermitian, positive semidefinite, trace 1).
- \(H(t)\) is the full-system Hamiltonian (drift + controls).
- \(C_j\) are collapse operators.

### 1.1 Hamiltonian model

The solver uses `ModelBase` terms:

- Drift: `model.drift_terms` are summed into a single cached drift matrix:
  \[
  H_\text{drift} = \sum_m H_m.
  \]

- Controls: `model.control_terms` contain pairs `(channel, H_k)`:
  \[
  H(t) = H_\text{drift} + \sum_k u_k(t)\, H_k.
  \]
  The coefficients \(u_k(t)\) come from the `PulseProgram`:
  - `u_k(t) = program.coeff(channel, t)`

### 1.2 Dissipators

Collapse operators come from `model.collapse_ops[j].C`.

For efficiency, the workspace precomputes:
- \(C_j^\dagger\)
- \(C_j^\dagger C_j\)

---

## 2. Numerics

### 2.1 Time discretization

The solver uses the `PulseProgram` time grid (`program.grid`) and advances \(\rho\) step-by-step between
consecutive grid points. For each interval:

- \(dt = t_{i+1} - t_i\)
- RK4 is applied to the matrix ODE

### 2.2 Sampling (`sample_stride`)

Expectations are recorded at grid indices:

- `0, stride, 2*stride, ...` and **always** the last grid point.

This matches the MCWF sampling convention used elsewhere in Phase 1.

### 2.3 Optional stabilizers

Numerical integration can drift slightly (trace/Hermiticity) due to finite precision. Two optional stabilizers exist:

- `enforce_trace_one`:
  - rescales \(\rho \leftarrow \rho / \mathrm{Tr}(\rho)\)

- `enforce_hermiticity`:
  - symmetrizes \(\rho \leftarrow (\rho + \rho^\dagger)/2\)

These are **debug/validation tools**, not strict physical projections (positivity is not enforced).

---

## 3. Data layout and conventions

### 3.1 Storage layout

The solver stores density matrices in **column-major** order as a flat array of length \(D^2\):

- index: `row + col * D`

This matches the storage convention used for `DenseOp::data()`.

### 3.2 Expectations

For each observable \(O\) (a `DenseOp` in the full Hilbert space), the solver records:

\[
\langle O \rangle(t) = \mathrm{Re}\,\mathrm{Tr}(O\,\rho(t)).
\]

(Imaginary parts should be approximately zero for Hermitian \(O\).)

---

## 4. API overview

### 4.1 Options

`fluxpulse::lindblad::LindbladOptions`:

- `sample_stride` (Dim, default 1): sampling stride on the grid.
- `enforce_trace_one` (bool, default true): renormalize by trace each step.
- `enforce_hermiticity` (bool, default false): symmetrize rho each step.

### 4.2 Workspace

`fluxpulse::lindblad::LindbladWorkspace` caches:

- drift sum `H_drift`
- `Cdag[j] = C_j^†` and `CdC[j] = C_j^† C_j`
- scratch buffers for RK4: `rho`, `k1..k4`, and temporary matrices

Call:
- `ws.init(model)` once per model (or whenever dimensions change)

### 4.3 Main entry point

```cpp
Status run_lindblad(const ModelBase& model,
                    const PulseProgram& program,
                    Span<const DenseOp> observables,
                    Span<const Complex> rho0,
                    const LindbladOptions& opt,
                    LindbladWorkspace* ws,
                    Result* out);
```

Inputs:
- `rho0`: flat \(D^2\) density matrix in column-major order

Outputs:
- `Result` with expectations sampled along the grid

### 4.4 Helper: build a pure-state density matrix

```cpp
Status rho_from_psi(Span<const Complex> psi, std::vector<Complex>* out_rho);
```

Produces \(\rho = |\psi\rangle\langle\psi|\) into `out_rho` (column-major).

---

## 5. Example usage (unit test / validation)

```cpp
#include "fluxpulse/sim/lindblad/lindblad_solver.hpp"
#include "fluxpulse/sim/mcwf/mcwf_solver.hpp"

// 1) Build model + program as usual
ModelBase model = ...;
PulseProgram program = ...;

// 2) Observables
std::vector<DenseOp> observables = ...;

// 3) Initial pure state -> density matrix
std::vector<Complex> psi0 = ...; // length D
std::vector<Complex> rho0;
fluxpulse::lindblad::rho_from_psi({psi0.data(), (Dim)psi0.size()}, &rho0);

// 4) Run Lindblad
fluxpulse::lindblad::LindbladWorkspace ws;
ws.init(model);

fluxpulse::lindblad::LindbladOptions opt;
opt.sample_stride = 1;

Result r_dm;
fluxpulse::lindblad::run_lindblad(model, program, {observables.data(), (Dim)observables.size()},
                                  {rho0.data(), (Dim)rho0.size()}, opt, &ws, &r_dm);

// 5) Compare against MCWF average (many trajectories)
Result r_mcwf = ...;
```

Recommended comparisons:
- max absolute error per observable: `max_t |⟨O⟩_mcwf(t) - ⟨O⟩_dm(t)|`
- expected scaling: ~ \(O(1/\sqrt{N_\text{traj}})\) for MCWF (stochastic error)

---

## 6. Performance and limitations

This solver is intentionally **dense** and **O(D^3)** per step (matrix multiplications).
It is meant for validation on small systems:

- 1 qubit: \(D=2\) (very fast)
- 2 qubits: \(D=4\) (fast)
- 3 qubits: \(D=8\) (still fine)
- 4–5 qubits: \(D=16\)–\(32\) (can become slow with large grids)

For production-scale multi-qubit simulations, MCWF is preferred.

---

## 7. Future refactors (optional)

If/when needed:
- Extract the RHS into `lindblad_rhs.hpp/.cpp` to allow:
  - testing the RHS in isolation
  - swapping integrators (RK4, RK45, etc.)
- Add positivity diagnostics (not projection) for debugging.
- Add a small “Liouvillian” builder for experiments (not required for Phase 1).

---

## 8. CMake integration reminder

Add `src/sim/lindblad/lindblad_solver.cpp` to the main library target sources so it links into tests/apps.
