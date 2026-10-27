# Two-Qubit Effective Model (Phase 1)

This document describes the **coupled-qubits effective model** implemented in:

- `include/fluxpulse/model/effective/coupled_qubits.hpp`
- `src/model/effective/coupled_qubits.cpp`

It focuses on the **Hamiltonian terms**, **noise terms**, and **conventions** (basis ordering, operator embedding, and channels).

---

## 1. Hilbert space and basis ordering

### 1.1 Subsystem dimensions
- Each qubit is a 2-level system, so the total Hilbert space dimension is:

\[
d = 2^{N}
\]

where \(N\) is the number of qubits.

### 1.2 Little-endian computational basis (important)
FluxPulse uses **little-endian** ordering:

- **Qubit 0** is the *least significant* bit (LSB).
- The computational basis state \(|q_{N-1}\dots q_1 q_0\rangle\) maps to the integer index:

\[
\text{index} = q_0 + 2 q_1 + 4 q_2 + \dots + 2^{N-1} q_{N-1}.
\]

Examples for **2 qubits**:
- \(|00\rangle\rightarrow 0\)
- \(|01\rangle\rightarrow 1\)  (q0=1, q1=0)
- \(|10\rangle\rightarrow 2\)  (q0=0, q1=1)
- \(|11\rangle\rightarrow 3\)

---

## 2. Operator embedding conventions

All model terms are built as **dense operators** (`DenseOp`) embedded into the full Hilbert space.

### 2.1 Single-qubit embedding
A local 2×2 operator \(A\) acting on qubit \(i\) is embedded using:

- `embed_qubit_op(A, i, N)`

This produces the full operator:

\[
I \otimes \cdots \otimes I \otimes A_i \otimes I \otimes \cdots \otimes I
\]

with the placement consistent with the **little-endian** basis.

### 2.2 Two-qubit product embedding
For a product operator \(A\) on qubit \(q_a\) and \(B\) on qubit \(q_b\), use:

- `embed_two_qubit_product(A, B, qa, qb, N)` (defined in `operators/tensor.hpp`)

**Important detail:** because of little-endian ordering, the helper constructs the local 4×4 as:

- `kron(op_hi, op_lo)` where `hi=max(qa,qb)` and `lo=min(qa,qb)`,
- then calls `embed_two_qubit_op(local4, lo, hi, N)`.

This guarantees that:
- “`A` acts on `qa` and `B` acts on `qb`”
is true regardless of whether `qa<qb` or `qa>qb`.

---

## 3. Pauli and ladder operators

Local single-qubit operators (2×2):

- \(X = \begin{pmatrix}0&1\\1&0\end{pmatrix}\)
- \(Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix}\)
- \(Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}\)
- \(\sigma_- = |0\rangle\langle 1| = \begin{pmatrix}0&1\\0&0\end{pmatrix}\)

---

## 4. Hamiltonian model

The coupled-qubits model builds:

\[
H(t) = H_\text{drift} + H_\text{ctrl}(t)
\]

### 4.1 Drift terms (per qubit)
For each qubit \(i\), the drift includes a Z term:

\[
H_{Z,i} = \tfrac{1}{2} (\omega_i + \Delta_i) Z_i
\]

- `omega[i]` corresponds to the qubit frequency parameter
- `detuning[i]` corresponds to a detuning parameter
- The **0.5 factor** matches the convention used in the single-qubit `GenericQubit` effective model.

If `omega` or `detuning` vectors are empty, they are treated as 0.

### 4.2 Control terms (per qubit)
Controls are added as *channelized* Hamiltonian terms with the same **0.5 factor** convention:

\[
H_\text{ctrl}(t) = \sum_i \Big(u_{x,i}(t)\,\tfrac{1}{2} X_i
+ u_{y,i}(t)\,\tfrac{1}{2} Y_i
+ u_{z,i}(t)\,\tfrac{1}{2} Z_i\Big).
\]

Each enabled axis contributes one `ControlTerm` per qubit, keyed by a `ChannelId`:

- Drive X: `0.5 * X_i`
- Drive Y: `0.5 * Y_i`
- Drive Z: `0.5 * Z_i`

> **Uniqueness constraint:** `ModelBase::validate()` requires **each `ChannelId` to be unique** across all control terms.
> Therefore, multi-qubit control requires distinct channel IDs per qubit and axis.

### 4.3 Default channel mapping (multi-qubit)
If explicit `drive_x/drive_y/drive_z` vectors are empty, default channels are generated:

- **DriveX** on qubit `i`: `ChannelId{ 1 + 3*i }`
- **DriveY** on qubit `i`: `ChannelId{ 2 + 3*i }`
- **DriveZ** on qubit `i`: `ChannelId{ 3 + 3*i }`

This matches the single-qubit convention for `i=0` (channels 1/2/3) and extends naturally.

If explicit channel vectors are provided, they must have size `nqubits` and:
- a qubit contributes a term only if the provided `ChannelId` is valid.

---

## 5. Two-qubit coupling terms

Coupling terms are appended to `drift_terms` as dense operators.

### 5.1 Exchange coupling (XY / iSWAP-style)
For each pair `(i,j)` with coefficient \(J\):

\[
H_{XY,ij} = J\,(X_i X_j + Y_i Y_j)
\]

This is useful for **SWAP oscillations** in the single-excitation manifold.

In code, this is built via:
- `embed_two_qubit_product(X, X, i, j, N)`
- `embed_two_qubit_product(Y, Y, i, j, N)`

then scaled by `J` and added as two drift terms.

### 5.2 ZZ coupling
For each pair `(i,j)` with coefficient \(\zeta\):

\[
H_{ZZ,ij} = \zeta\, Z_i Z_j
\]

This term is useful for **ZZ phase accumulation**.

---

## 6. Noise model (collapse operators)

The coupled model optionally adds per-qubit collapse operators using the same convention as `GenericQubit`:

### 6.1 T1 (amplitude damping)
If `t1[i] > 0`, define:

\[
\gamma_{1,i} = \frac{1}{T_{1,i}}
\]

Collapse operator:

\[
C_{1,i} = \sqrt{\gamma_{1,i}}\, \sigma_{-,i}
\]

In code, the local 2×2 operator is built using:
- `noise::qubit_amplitude_damping(gamma1, &C2)`
then embedded with `embed_qubit_op(C2, i, N)`.

### 6.2 Tphi (pure dephasing)
If `tphi[i] > 0`, define:

\[
\gamma_{\phi,i} = \frac{1}{T_{\phi,i}}
\]

Collapse operator:

\[
C_{\phi,i} = \sqrt{\frac{\gamma_{\phi,i}}{2}}\, Z_i
\]

Built using:
- `noise::qubit_pure_dephasing(gamma_phi, &C2)`
then embedded with `embed_qubit_op(C2, i, N)`.

---

## 7. Units and scaling

FluxPulse treats coefficients consistently with the solver’s time unit:

- If time is in seconds, frequencies/couplings are typically in **rad/s**.
- If you store frequencies in **Hz**, convert to rad/s using \(2\pi\) before building the model.

The code does not enforce a specific unit system; it assumes all parameters are self-consistent.

---

## 8. Parameter summary (`CoupledQubitsParams`)

- `nqubits`: number of qubits (>=1)
- `omega[i]`, `detuning[i]`: drift Z parameters (optional)
- `enable_drive_{x,y,z}`: global flags
- `drive_{x,y,z}[i]`: optional explicit channels (must be unique if used)
- `exchange_xy`: list of `{q0,q1,strength}` for \(J(XX+YY)\)
- `zz`: list of `{q0,q1,strength}` for \(\zeta ZZ\)
- `t1[i]`, `tphi[i]`: optional noise times

The builder performs:
- size checks (vectors empty or length `nqubits`)
- coupling index checks
- `ModelBase::validate()` (including channel ID uniqueness)

---

## 9. Practical examples

### 9.1 2Q SWAP oscillations
- Set `exchange_xy = [{q0=0,q1=1,strength=J}]`
- Set `zz` empty (or 0)
- No drives required
- Initialize \(|10\rangle\) and track populations \(P_{10}, P_{01}\)

### 9.2 ZZ phase accumulation
- Set `zz = [{q0=0,q1=1,strength=zeta}]`
- Set `exchange_xy` empty
- Prepare a superposition state and track phase-sensitive observables (e.g. overlap or correlators)

---

## 10. Related implementation files

- Model builder:
  - `include/fluxpulse/model/effective/coupled_qubits.hpp`
  - `src/model/effective/coupled_qubits.cpp`
- Tensor / embedding utilities:
  - `include/fluxpulse/operators/tensor.hpp`
- Noise collapse op builders:
  - `include/fluxpulse/model/noise/collapse_ops.hpp`
  - `src/model/noise/collapse_ops.cpp` (if present)

