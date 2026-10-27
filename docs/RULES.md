# FluxPulse Rules

This document defines project rules for repository hygiene, **C-like C++** coding style, build/test discipline, determinism, and contribution workflow.

The goal is: **write C++ that reads like careful C**, while staying CUDA-ready and BLAS-ready by keeping computation kernels pointer/view-based and isolating ownership/allocation/backends.

---

## 0. C-like C++ policy (read this first)

### 0.1 Allowed C++ (keep it boring)
Prefer:
- `struct` with public fields for POD-like data types
- RAII for ownership (containers/buffers free themselves)
- `constexpr`, `enum class`, `static_cast`, `nullptr`
- Minimal, explicit interfaces where they buy clarity (e.g., `Op::apply()`)

Standard library usage is allowed **primarily for ownership and basic utilities**:
- `std::vector`, `std::string`, `std::optional`

**Important**: kernels/operators must not *require* std containers; they must operate on raw pointers or `Span<T>`.

### 0.2 Avoid (project-wide)
Avoid in FluxPulse code:
- Template metaprogramming (no SFINAE, no heavy `concepts`, no type-traits tricks)
- Complicated inheritance (no deep hierarchies, no multiple inheritance, no mixins)
- Exceptions (`throw`, `try/catch` in project code)
- Operator overloading beyond basic indexing/comparison
  - OK: `operator[]` for container-like access; `==/!=` for simple comparisons
  - Avoid: linear-algebra DSL overloads (`+`, `*`, etc.)

---

## 1. Repository hygiene

### 1.1 No generated files in git
Do **not** commit:
- `build*/` (out-of-source build dirs)
- CMake-generated files
- compiler outputs (`*.o`, `*.a`, `*.so`, `*.exe`, etc.)
- IDE folders (`.vscode/`, `.idea/`)
- logs / temp files

### 1.2 Header-only exceptions
Default style is “header + source”, but **small leaf utilities** may be header-only to reduce file sprawl and support CUDA compilation later.

Header-only (Phase 1):
- `include/fluxpulse/core/types.hpp` (defines `FLUXPULSE_HD`)
- `include/fluxpulse/core/span.hpp`
- `include/fluxpulse/core/rng.hpp`
- `include/fluxpulse/core/timer.hpp`
- `include/fluxpulse/math/real_math.hpp`
- `include/fluxpulse/math/complex.hpp`
- small operator constructors under `include/fluxpulse/operators/` (e.g., `pauli.hpp`, `ladder.hpp`, `tensor.hpp`)

If a header-only file grows large or changes often, split it later.

---

## 2. Error handling (no exceptions)

### 2.1 Rules
- FluxPulse code must not `throw`.
- Do not use `try/catch` in FluxPulse code.
- Prefer explicit error returns (bool/status + error string) for I/O and config.

### 2.2 Preconditions
For math kernels and tight loops:
- Document preconditions.
- Use `assert(...)` in Debug builds.

---

## 3. Ownership and memory

### 3.1 Default ownership
- Use `std::vector<T>` for owned CPU buffers.
- Avoid raw `new/delete` outside dedicated allocators/wrappers.
- Avoid `shared_ptr` unless absolutely needed.
- `unique_ptr` only for polymorphic ownership or pimpl.

### 3.2 Allocation discipline
- No per-step allocations inside solver inner loops.
- Pre-allocate and reuse buffers.

---

## 4. Build rules (CMake)

- Out-of-source builds only.
- Target standard: C++20.
- Keep warning-clean.

Recommended (when possible):
- compile with exceptions disabled (`-fno-exceptions`), while still avoiding exceptions in code even if enabled.

---

## 5. Determinism and randomness

### 5.1 Reproducibility is mandatory
Given the same config and base seed, each global trajectory ID must use the same random stream across MPI sizes. Ensemble reductions can differ in floating-point rounding when rank count or scheduling changes.

### 5.2 Seeding rules
- All randomness must go through `core/rng.hpp`.
- Use deterministic per-trajectory seeding:
  - `derive_trajectory_seed(base_seed, global_trajectory_id)`
- RNG must be swappable via `RngBackend` (function pointer table), but default behavior must remain deterministic.

---

## 6. CUDA-readiness rules (MANDATORY)

### 6.1 Kernel signatures are pointer/view based
All kernels and `Op::apply`-like functions must accept:
- raw pointers + sizes, or
- `Span<T>` / `Span<const T>`

Do not write kernels that require iterators or `std::vector` as input.

### 6.2 Complex numbers
- Kernels must use `fluxpulse::Complex` POD type (not `std::complex`).
- Avoid `<cmath>` in kernel-facing headers; route through `math/real_math.hpp`.

### 6.3 `FLUXPULSE_HD`
Use `FLUXPULSE_HD` for tiny math helpers that may later compile for GPU. It expands to `__host__ __device__` under NVCC and is empty otherwise.

---

## 7. BLAS readiness rules (MANDATORY)

### 7.1 Dense layout
`DenseOp` stores matrices in **column-major** order and exposes:
- `data()` (pointer to contiguous storage)
- `lda()` (leading dimension)

### 7.2 BLAS-shaped API
`DenseOp` must provide:
- `gemv(y, x, alpha, beta)` implementing `y = alpha*A*x + beta*y`

Solvers should call `Op::apply` / `gemv` and must not depend on BLAS directly.

### 7.3 Centralized BLAS wrapper
All BLAS/cuBLAS headers, handles, and type conversions must live in a dedicated wrapper:
- `linalg/blas_gemv.hpp/.cpp` (naive today; CBLAS/cuBLAS later)

No vendor complex types should appear in `DenseOp` or solver code.

---

## 8. Operator and basis conventions (MANDATORY)

### 8.1 Subsystem ordering
FluxPulse uses **little-endian subsystem ordering**:
- subsystem 0 is the least significant (fastest-varying digit)
- for qubits: `|q_{N-1}...q_1 q_0⟩` maps to `index = q0 + 2*q1 + 4*q2 + ...`

### 8.2 Kronecker product and embedding
- `kron(A, B)` means `A` acts on the more significant subsystem(s) and `B` on the least significant.
- Embedding utilities in `operators/tensor.hpp` must follow §8.1.

---

## 9. Testing rules

### 9.1 Smoke tests
Phase 1 starts with **standalone assert-based smoke tests** (executables).
- They should be deterministic.
- They should validate conventions (layout/order) explicitly.

### 9.2 Avoid clang volatile warnings (Apple toolchains)
Avoid `volatile_var += x` in tests; use `volatile_var = volatile_var + x` to prevent `-Wdeprecated-volatile`.

---

## 10. Documentation rules
- Architecture and convention changes must update `DESIGN.md`.
- New core design choices must be recorded (layout, ordering, determinism policy, etc.).

---

## 11. Contribution workflow
- Keep commits focused.
- Format consistently.
- Before merging: Debug+Release builds, tests pass, docs updated when needed.
