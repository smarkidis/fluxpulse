# Monte Carlo Wave Function (MCWF) Solver in FluxPulse

FluxPulse Phase 1 uses the **Monte Carlo Wave Function (MCWF)** method to simulate open quantum systems described by a Lindblad master equation, but using **stochastic pure-state trajectories**.

This document explains the core ideas and how the solver is implemented in the framework.

---

## 1) Motivation: open systems without a density matrix

The Lindblad master equation evolves a density matrix \(\rho\) of size \(d\times d\):

\[
\frac{d\rho}{dt} = -i[H(t),\rho] + \sum_n \left(C_n\rho C_n^{\dagger} - \frac{1}{2}\{C_n^{\dagger}C_n,\rho\}\right)
\]

Computing \(\rho\) directly costs \(O(d^2)\) memory and \(O(d^3)\) time for many operations.

MCWF replaces this with an average over **state vectors** \(|\psi\rangle\) of size \(d\), which can be much cheaper when you only need expectation values.

---

## 2) Core idea of MCWF

MCWF represents the density matrix as an ensemble average over trajectories:

\[
\rho(t) \approx \frac{1}{N}\sum_{j=1}^{N} |\psi_j(t)\rangle\langle\psi_j(t)|
\]

Each trajectory evolves mostly deterministically, but occasionally experiences **quantum jumps**.

---

## 3) Effective (non-Hermitian) Hamiltonian

Between jumps, the state evolves under an **effective Hamiltonian**:

\[
H_{\mathrm{eff}}(t) = H(t) - \frac{i}{2}\sum_n C_n^{\dagger}C_n
\]

where:

\[
H(t) = H_0 + \sum_k u_k(t) H_k
\]

- \(H_0\): drift Hamiltonian terms
- \(H_k\): fixed control Hamiltonian terms
- \(u_k(t)\): time-dependent classical control coefficients from the pulse program
- \(C_n\): collapse operators describing noise

The Schrödinger-like equation is:

\[
\frac{d|\psi\rangle}{dt} = -i H_{\mathrm{eff}}(t)|\psi\rangle
\]

Because \(H_{\mathrm{eff}}\) is non-Hermitian, this evolution **reduces the norm** of \(|\psi\rangle\). That norm loss is not an error—it encodes the probability that a jump should occur.

---

## 4) Jump probabilities

For each collapse operator \(C_n\), the jump probability over a small step \(dt\) is:

\[
p_n = dt\,\langle\psi|C_n^{\dagger}C_n|\psi\rangle
    = dt\,\|C_n|\psi\rangle\|^2
\]

Total jump probability:

\[
p_{\mathrm{tot}} = \sum_n p_n
\]

In practice:
- draw a uniform random number \(r\in[0,1)\)
- if \(r < p_{\mathrm{tot}}\), a jump happens
- otherwise, no jump

> If \(p_{\mathrm{tot}}\) becomes large (e.g. \(> 0.1\)), you should reduce `dt` for accuracy.

---

## 5) Selecting which jump occurs

If a jump occurs, select index \(n\) with probability:

\[
\Pr(n) = \frac{p_n}{p_{\mathrm{tot}}}
\]

Implementation-wise:
- draw another uniform random \(r_2\in[0,1)\)
- set threshold \(r_2 p_{\mathrm{tot}}\)
- accumulate \(p_n\) until exceeding threshold

---

## 6) Applying a jump

When jump \(n\) occurs, update the state:

\[
|\psi\rangle \leftarrow \frac{C_n|\psi\rangle}{\|C_n|\psi\rangle\|}
\]

This is a non-unitary “reset” event (e.g., relaxation from \(|1\rangle\to|0\rangle\) for a qubit).

---

## 7) What happens if no jump occurs?

If no jump occurs:
1) accept the deterministic evolution step under \(H_{\mathrm{eff}}\)
2) renormalize the state

Renormalization is recommended for numerical stability, even though the non-Hermitian evolution encodes probability information.

---

## 8) How FluxPulse implements MCWF (Phase 1)

### Inputs
The solver consumes:

- `ModelBase`:
  - drift terms \(H_0\)
  - control terms \(H_k\) keyed by channel
  - collapse operators \(C_n\)

- `PulseProgram`:
  - `TimeGrid`
  - per-channel envelopes to compute \(u_k(t)\)

- initial state \(|\psi_0\rangle\)

- a deterministic RNG seeded per trajectory

### Precomputation
Before stepping:
- compute \(K = \sum_n C_n^{\dagger}C_n\) once (if collapse operators exist)
- allocate work buffers in `McwfWorkspace`
- create a `PulseProgram::Cursor` to evaluate envelopes efficiently in monotonic time loops

### Per-step loop (conceptual)
For each grid interval \([t_i, t_{i+1}]\) with step \(dt\):

1) **Compute jump probabilities** from the *current state*:
   - for each collapse op: compute \(C_n|\psi\rangle\), then \(p_n = dt\,\|C_n\psi\|^2\)

2) **Deterministic propagation** with RK4 under \(H_{\mathrm{eff}}(t)\):
   - build \(H(t)\psi\) from drift + Σ u_k(t) H_k
   - compute \(-iH(t)\psi - 0.5 K\psi\)

3) **Stochastic decision**:
   - if jump occurs: apply jump operator to current state and renormalize
   - else: accept RK4 result and renormalize

4) **Measure observables** (optional):
   - compute \(\langle O \rangle = \langle\psi|O|\psi\rangle\) for configured observables
   - record in `Result` or accumulate into sums for averaging

### Multiple trajectories
To estimate density-matrix expectations:
- run many trajectories with different RNG streams
- average observables across trajectories

---

## 9) Determinism and reproducibility

FluxPulse aims for reproducible MCWF results when:
- base seed is fixed
- trajectory index is fixed
- code version is fixed

Typical orchestration:
- `seed_traj = hash(base_seed, traj_index)`
- each trajectory uses its own RNG stream

---

## 10) Practical accuracy considerations

- **Time step size (`dt`)**: must be small enough that jump probabilities per step remain small, and RK4 remains stable.
- **Trajectory count**: statistical error scales roughly as \(1/\sqrt{N}\).
- **Observables**: expectation values converge faster than full distributions; start with Z-population or number operator.

---

## Summary

MCWF simulates Lindblad dynamics by averaging over stochastic state-vector trajectories.

Key components:
- **Effective Hamiltonian**: \(H_{\mathrm{eff}} = H - \tfrac{i}{2}\sum C^{\dagger}C\)
- **Jump probabilities**: \(p_n = dt\,\|C_n\psi\|^2\)
- **Jump update**: \(|\psi\rangle \leftarrow C_n|\psi\rangle / \|C_n\psi\|\)
- **Trajectory averaging** reproduces open-system expectation values.

In FluxPulse, `ModelBase` provides operators, `PulseProgram` provides time-dependent coefficients, RK4 integrates the effective equation, and jump utilities implement the stochastic events.
