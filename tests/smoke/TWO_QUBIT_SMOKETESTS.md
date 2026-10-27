# Two-Qubit Smoke Tests (Phase 1)

This document describes the two new smoke tests and their companion configs:

- `tests/smoke/smoketest_2q_swap.cpp` + `configs/2q_swap.json`
- `tests/smoke/smoketest_2q_zz.cpp` + `configs/2q_zz.json`

Both tests are **deterministic** (1 trajectory, fixed seed) and use assert-based checks consistent with the existing 1-qubit smoke tests.

---

## 1. Common conventions

### 1.1 Basis ordering (little-endian)
The 2-qubit computational basis is ordered as:

\[
|q_1 q_0\rangle \equiv [|00\rangle, |01\rangle, |10\rangle, |11\rangle]
\]

So the linear indices are:
- \(|00\rangle \to 0\)
- \(|01\rangle \to 1\)
- \(|10\rangle \to 2\)
- \(|11\rangle \to 3\)

### 1.2 Observable naming in tests
The tests build observables directly as `DenseOp`:

- **Projectors**
  - `P10` = \(|10\rangle\langle 10|\) → diagonal at index 2
  - `P01` = \(|01\rangle\langle 01|\) → diagonal at index 1
- **Single-qubit X operators**
  - `X0` acts on **qubit 0** (LSB): couples \(|00\rangle\leftrightarrow|01\rangle\) and \(|10\rangle\leftrightarrow|11\rangle\)
  - `X1` acts on **qubit 1** (MSB): couples \(|00\rangle\leftrightarrow|10\rangle\) and \(|01\rangle\leftrightarrow|11\rangle\)

---

## 2. Smoke test: SWAP oscillations

### 2.1 Files
- `tests/smoke/smoketest_2q_swap.cpp`
- `configs/2q_swap.json`

### 2.2 Model
Hamiltonian (drift only):
\[
H = J (X_0 X_1 + Y_0 Y_1)
\]

Noise: none.

Controls: none (no pulses).

### 2.3 Initial state
The test starts in:
\[
|10\rangle
\]
which is basis index **2** in the ordering above.

### 2.4 Observables and expected behavior
The test tracks:
- \(P_{10}(t)\)
- \(P_{01}(t)\)

In the single-excitation subspace \(\{|01\rangle, |10\rangle\}\), the exchange Hamiltonian produces SWAP oscillations:

\[
P_{01}(t) = \sin^2(2 J t), \quad P_{10}(t) = \cos^2(2 J t).
\]

Key checkpoint times:
- **t = 0**: \(P_{10} \approx 1\), \(P_{01} \approx 0\)
- **t\_swap = \pi/(4J)**: \(P_{01} \approx 1\), \(P_{10} \approx 0\)

The smoke test checks both the exact endpoints and the analytic value (within a smoke-level tolerance).

### 2.5 Config highlights (`configs/2q_swap.json`)
- `model.type = "coupled_qubits"`
- `exchange_xy = [{q0:0, q1:1, strength:J}]`
- no pulses
- `observables = ["P:10", "P:01"]` (for result output)
- output writes:
  - `out/2q_swap_result.json`
  - `out/2q_swap_trace.json`

---

## 3. Smoke test: ZZ phase accumulation

### 3.1 Files
- `tests/smoke/smoketest_2q_zz.cpp`
- `configs/2q_zz.json`

### 3.2 Model
Hamiltonian (drift only):
\[
H = \zeta\, Z_0 Z_1
\]

Noise: none.

Controls: none in the smoke test (the test prepares \(|++\rangle\) directly as an initial state vector).

> Note: The JSON config includes optional Y-drive pulses (to prepare \(|+\rangle\) states) but the smoke test itself uses a direct state-vector initialization for robustness.

### 3.3 Initial state
The test starts in:
\[
|++\rangle = \frac{1}{2} (|00\rangle + |01\rangle + |10\rangle + |11\rangle)
\]
i.e. all amplitudes are \(1/2\).

### 3.4 Observables and expected behavior
The test tracks:
- \(\langle X_0 \rangle(t)\)
- \(\langle X_1 \rangle(t)\)

Under pure ZZ evolution from \(|++\rangle\), the single-qubit X expectations oscillate as:

\[
\langle X_0 \rangle(t) = \langle X_1 \rangle(t) = \cos(2 \zeta t).
\]

Key checkpoint times:
- **t = 0**: \(\langle X \rangle \approx 1\)
- **t = \pi/(4\zeta)**: \(\cos(\pi/2)=0\)
- **t = \pi/(2\zeta)**: \(\cos(\pi)=-1\)

The smoke test checks the expected values (within a smoke-level tolerance) at these times.

### 3.5 Config highlights (`configs/2q_zz.json`)
- `model.type = "coupled_qubits"`
- `zz = [{q0:0, q1:1, strength:zeta}]`
- `observables = ["X0", "X1"]`
- output writes:
  - `out/2q_zz_result.json`
  - `out/2q_zz_trace.json`

---

## 4. How to run

### 4.1 Build + run smoke tests with CTest
From your build directory:

```bash
ctest -R smoketest_2q_swap -VV
ctest -R smoketest_2q_zz -VV
```

Or run all smoke tests:

```bash
ctest -R smoketest_ -VV
```

### 4.2 Run via config runner (if you use an app runner)
If your project has a config-based runner (example name):

```bash
./fluxpulse_run configs/2q_swap.json
./fluxpulse_run configs/2q_zz.json
```

The JSON configs are meant as **end-to-end examples** for the IO pipeline and result/trace outputs.

---

## 5. Debugging tips

- If SWAP doesn’t swap:
  - verify basis ordering is \(|q_1 q_0\rangle\)
  - verify exchange term is implemented as **XX + YY** (not XX alone)
  - check that no drift terms are accidentally enabled (omega/detuning set to 0)
- If ZZ oscillation is wrong sign/frequency:
  - verify the ZZ term is \(\zeta Z_0 Z_1\)
  - verify the expected formula \(\cos(2\zeta t)\) matches your Hamiltonian convention

---

## 6. Summary

These smoke tests provide quick Phase-1 validation that:

- **2-qubit coupling operators are embedded correctly** (little-endian).
- The MCWF pipeline produces correct unitary evolution when noise is disabled.
- Basic 2Q physics targets are hit:
  - exchange-driven SWAP oscillations
  - ZZ-driven phase accumulation observable in \(\langle X \rangle\).
