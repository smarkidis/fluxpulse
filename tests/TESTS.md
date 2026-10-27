# FluxPulse Smoke Tests

This document explains the initial smoke tests included in Phase 1.  
These tests are **standalone executables** (no gtest required) and use `assert()` for verification.

They are intended to catch:
- include/link issues,
- broken conventions (layout/order),
- basic correctness regressions,
- deterministic RNG mistakes.

---

## Tests overview

### 1) `smoketest_rng_timer.cpp`

**Purpose:** Validate the “leaf” utilities that everything else depends on:
- RNG reproducibility and API shape
- Timer monotonic behavior and basic timing calls

**What it checks**
- **SplitMix64 reproducibility**
  - Two RNG instances seeded with `derive_trajectory_seed(base_seed, traj_id)` must produce identical sequences.
- **Uniform sampling bounds**
  - `uniform01()` returns values in `[0, 1)`
  - `uniform01_open()` returns values in `(0, 1)`
- **Integer range sampling**
  - `uniform_u64(lo, hi)` stays inside the inclusive range `[lo, hi]`
- **External RNG backend plumbing**
  - `RngBackend` function pointer table works correctly (seed + next_u64)
- **Timer sanity**
  - `elapsed_seconds()` and `elapsed_ns()` are non-negative and monotonic
  - `now_ns()` is non-decreasing

**Notes**
- On Apple Clang, compound assignment to `volatile` is deprecated (`sink += i`). The test avoids warnings by using `sink = sink + i`.

---

### 2) `smoketest_core_linalg_operators.cpp`

**Purpose:** Validate core math + operator infrastructure used by Phase 1 simulation:
- `Span<T>` and pointer/view conventions
- POD `Complex` math
- `Vec` kernels
- `DenseOp` + BLAS-shaped `gemv`
- Pauli / ladder / tensor embedding conventions

**What it checks**
- **Span basics**
  - pointer + size correctness
  - read/write access via `operator[]`
- **Real math wrappers**
  - `real_sqrt`, `real_abs`, `real_exp`, `real_log`, `real_sin`, `real_cos`
- **Complex arithmetic**
  - `add`, `mul`, `abs2`, `abs`
- **Deterministic RNG sanity**
  - quick reproducibility check
- **Vec kernels**
  - `vec_copy`, `vec_set`, `vec_scale_inplace`, `vec_axpy_inplace`, `vec_dotc`, `vec_norm2`, `vec_normalize_inplace`
- **DenseOp correctness**
  - column-major memory layout sanity check (`data()[col*lda + row]`)
  - `apply(out, in)` equals `gemv(out, in, alpha=1, beta=0)`
  - `gemv(alpha, beta)` behavior using a known 2×2 matrix
- **Pauli operators**
  - `X|0⟩ = |1⟩`
  - `Z|1⟩ = -|1⟩`
  - `sigma_minus|1⟩ = |0⟩`
- **Ladder operators**
  - `a|2⟩ = sqrt(2)|1⟩`
  - `a†|2⟩ = sqrt(3)|3⟩`
  - `n|2⟩ = 2|2⟩`
- **Tensor conventions**
  - **Little-endian subsystem ordering**: qubit 0 is least significant
    - for 2 qubits: `|q1 q0⟩` maps to index `q0 + 2*q1`
  - `kron(I, X)` flips `q0`
  - `kron(X, I)` flips `q1`
  - `embed_qubit_op(X,0,2)` matches `kron(I,X)`
  - embedding test on non-qubit dims `[2,3,2]` verifies correct digit mapping

**Important gotcha (const spans)**
Your current `Span<T>` does **not** implicitly convert to `Span<const T>`.  
So, the tests explicitly create const spans for inputs:
- `make_span<const Complex>(vec.data(), vec.size())`

If you later add either:
- `Vec::cview()` returning `Span<const Complex>`, or
- a converting constructor in `Span`
you can simplify call sites.

---

## How to build & run (CMake)

Assuming you added both executables in `CMakeLists.txt`:

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Run tests via CTest

Show output on failure:
```bash
ctest --test-dir build --output-on-failure
```

Show full output (verbose):
```bash
ctest --test-dir build -VV
```

Run a single test:
```bash
ctest --test-dir build -R smoketest_rng_timer -VV
ctest --test-dir build -R smoketest_core_linalg_operators -VV
```

### Run executables directly
```bash
./build/smoketest_rng_timer
./build/smoketest_core_linalg_operators
```

---

## When to add more tests

As you move into Phase 1 solver work, add:
- RK4 step tests (toy linear ODE, compare to known solution)
- Effective Hamiltonian builder tests (`H_eff = H - i/2 Σ L†L`)
- Jump sampling tests (fixed seed, known cumulative rates)
- Statistical ensemble checks (T1 decay vs expected curve within tolerance)
- MPI consistency test (same base seed & trajectory count, invariant across rank counts)
