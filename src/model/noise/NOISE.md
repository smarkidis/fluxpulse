# Noise in FluxPulse (Phase 1)

FluxPulse models dissipation and decoherence using **Lindblad collapse operators** and simulates trajectories using the **Monte Carlo Wave Function (MCWF)** method.

This document describes how “noise” is represented in the framework and what conventions are used in Phase 1.

---

## 1) What “noise” means here

In this simulator, **noise** refers to *non-unitary* open-system effects such as:

- **Energy relaxation** (T1 / amplitude damping)
- **Pure dephasing** (Tϕ)
- (later) photon loss for oscillators, thermal excitation, etc.

These effects are represented through a Markovian master equation in Lindblad form.

---

## 2) Lindblad master equation representation

The density matrix evolution is:

\[
\frac{d\rho}{dt} = -i [H(t), \rho] + \sum_n \left( C_n \rho C_n^{\dagger} - \frac{1}{2}\{C_n^{\dagger} C_n, \rho\} \right)
\]

- \(H(t)\) is the Hamiltonian (drift + controls).
- Each \(C_n\) is a **collapse operator** representing one dissipation channel.

In FluxPulse, a model provides:
- drift terms \(H_0\)
- control terms \(H_k\)
- a list of collapse operators \(\{C_n\}\)

---

## 3) MCWF (trajectory) view of noise

MCWF simulates the same Lindblad evolution by averaging over **stochastic pure-state trajectories**.

At each time step \(dt\):

1) Evolve the state under the **effective non-Hermitian Hamiltonian**:
\[
H_{\mathrm{eff}}(t) = H(t) - \frac{i}{2}\sum_n C_n^{\dagger}C_n
\]

2) Compute **jump probabilities**:
\[
p_n = dt\,\langle\psi|C_n^{\dagger}C_n|\psi\rangle
\]

3) With probability \(p = \sum_n p_n\), apply a quantum jump:
\[
|\psi\rangle \leftarrow \frac{C_n |\psi\rangle}{\|C_n |\psi\rangle\|}
\]
Otherwise renormalize after the non-Hermitian evolution.

Averaging observables over many trajectories approximates the density matrix solution.

---

## 4) Noise objects in code

### CollapseOp
A collapse operator is represented as a matrix (`DenseOp`) stored inside:

- `CollapseOp { DenseOp C; }`

and models store a list of these operators.

### Helper constructors
Phase 1 includes helper functions in `fluxpulse/model/noise/collapse_ops.*` to create common collapse operators for qubits.

---

## 5) Phase 1 collapse operators and conventions

### A) Amplitude damping (T1)
Amplitude damping models energy relaxation from \(|1\rangle \to |0\rangle\).

Collapse operator:
\[
C = \sqrt{\gamma_1}\,\sigma_-,
\quad \sigma_- = |0\rangle\langle 1|
\]

Rate:
\[
\gamma_1 = 1/T_1
\]

This yields an exponential decay of excited-state population with rate \(\gamma_1\).

### B) Pure dephasing (Tϕ)
Pure dephasing reduces coherence without changing populations.

There are multiple equivalent collapse-operator conventions for dephasing.  
In Phase 1 FluxPulse uses:

\[
C = \sqrt{\gamma_\phi/2}\,\sigma_z
\]

with:
\[
\gamma_\phi = 1/T_\phi
\]

This factor of 1/2 is chosen so that the off-diagonal coherence decay rate matches \(\gamma_\phi\) under the standard Lindblad form.

> Important: The simulator must be internally consistent; if you change this convention, update tests accordingly.

---

## 6) Where noise enters the simulation pipeline

Noise is injected by the **model**, not by the pulse program.

- The pulse program provides time-dependent coefficients \(u_k(t)\).
- The model provides:
  - Hamiltonian operators \(H_0\), \(H_k\)
  - collapse operators \(C_n\)

The MCWF solver uses both:
- builds \(H(t)\) from drift + controls
- forms \(H_{\mathrm{eff}}\) using \(C_n^{\dagger}C_n\)
- performs stochastic jumps

---

## 7) Multi-qubit and multi-level extensions (later)

The same abstraction generalizes to:

- multi-qubit relaxation/dephasing: collapse operators embedded into a larger Hilbert space
- oscillator photon loss: \(C = \sqrt{\kappa}\,a\)
- thermal excitation: \(C = \sqrt{\kappa n_{\mathrm{th}}}\,a^{\dagger}\)

These are Phase 2+ additions but fit into the same `collapse_ops` list in `ModelBase`.

---

## 8) Practical notes

- **Determinism**: noise is stochastic but should be reproducible for a given base seed + trajectory index.
- **Performance**: compute/store \(C_n^{\dagger}C_n\) once per model if possible (not per time step).
- **Validation**: sanity tests (T1 decay curve, dephasing coherence decay) are critical anchors for correctness.

---

## Summary

FluxPulse represents noise using **collapse operators** \(C_n\) in Lindblad form.  
Phase 1 focuses on:
- **T1** amplitude damping: \(\sqrt{\gamma_1}\,\sigma_-\)
- **Tϕ** pure dephasing: \(\sqrt{\gamma_\phi/2}\,\sigma_z\)

The MCWF solver uses these operators to form the effective Hamiltonian and stochastic jumps, and trajectory averaging reproduces open-system dynamics.
