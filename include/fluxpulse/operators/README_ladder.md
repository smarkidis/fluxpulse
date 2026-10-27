# ladder.hpp — Ladder Operators (Truncated Oscillator Basis)

`include/fluxpulse/operators/ladder.hpp` provides helper constructors that build common ladder operators as dense matrices (`fluxpulse::DenseOp`) for a **finite-dimensional truncated harmonic oscillator basis**.

These operators are used for oscillator, resonator, Kerr-oscillator, and transmon-like effective models where the Hilbert space is truncated to dimension `N`.

---

## Basis convention

We use the standard number (Fock) basis:

- Basis states: \|k⟩ for `k = 0, 1, ..., N-1`
- Truncation dimension: `N >= 1`

---

## Operators provided

All constructors return a **square dense operator** (`DenseOp`) of size `N x N` in **column-major layout** (see `linalg/dense_op.hpp`).

### 1) Annihilation operator `a`

**Definition:**
- `a |k⟩ = sqrt(k) |k-1⟩`
- with `a |0⟩ = 0`

**Non-zero matrix elements:**
- `a_{k-1, k} = sqrt(k)` for `k = 1..N-1`

**API:**
```cpp
fluxpulse::DenseOp fluxpulse::annihilation(fluxpulse::Dim N);
```

---

### 2) Creation operator `a†`

**Definition:**
- `a† |k⟩ = sqrt(k+1) |k+1⟩`
- with `a† |N-1⟩` truncated to 0

**Non-zero matrix elements:**
- `a†_{k+1, k} = sqrt(k+1)` for `k = 0..N-2`

**API:**
```cpp
fluxpulse::DenseOp fluxpulse::creation(fluxpulse::Dim N);
```

---

### 3) Number operator `n = a† a`

**Definition:**
- `n |k⟩ = k |k⟩`

**Matrix:**
- diagonal: `diag(0, 1, 2, ..., N-1)`

**API:**
```cpp
fluxpulse::DenseOp fluxpulse::number_op(fluxpulse::Dim N);
```

---

### 4) Projector `|k⟩⟨k|`

A diagonal projector onto a single Fock state.

**API:**
```cpp
fluxpulse::DenseOp fluxpulse::projector(fluxpulse::Dim N, fluxpulse::Dim k);
```

---

## Storage layout notes (important)

`DenseOp` is stored in **column-major** order:

- element `(row, col)` is stored at `data()[col * lda + row]`
- in these helpers, you should always use `op.set(row, col, value)` rather than indexing raw storage.

---

## Example usage

```cpp
#include "fluxpulse/operators/ladder.hpp"
#include "fluxpulse/linalg/vec.hpp"

using namespace fluxpulse;

int main() {
  const Dim N = 5;

  DenseOp a  = annihilation(N);
  DenseOp ad = creation(N);
  DenseOp n  = number_op(N);

  Vec psi(N);
  psi.set_zero();
  psi[2] = Complex{1.0, 0.0}; // |2>

  Vec out(N);
  out.set_zero();

  // out = a * |2> = sqrt(2) |1>
  a.apply(out.view(), psi.view());

  return 0;
}
```

---

## Design goals

- **C-like C++:** no exceptions, minimal abstractions.
- **CUDA-ready:** uses `fluxpulse::Complex` (POD) and pointer/view-based `apply()` in the operator layer.
- **BLAS-ready:** Dense operators use GEMV-shaped APIs internally.

---

## Limitations

- Operators are constructed as **dense matrices**, so memory scales as `O(N^2)`.
- Truncation is explicit: creation operator is truncated at the top level.

---

## Related files

- `include/fluxpulse/operators/pauli.hpp` — 2×2 Pauli operators for qubits
- `include/fluxpulse/linalg/dense_op.hpp` — dense operator storage and GEMV interface
- `include/fluxpulse/linalg/op.hpp` — operator apply interface
- `include/fluxpulse/math/complex.hpp` — POD complex type
