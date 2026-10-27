# Generic Qubit Effective Model (Phase 1)

FluxPulse Phase 1 includes a minimal **generic 2-level qubit** model intended to support early validation tests:
- Rabi
- Ramsey
- T1 decay (and optionally pure dephasing via Tϕ)

This document explains the model, its parameters, and how it maps into the simulator abstractions.

---

## 1) What this model represents

The model describes a single qubit with Hilbert space dimension:

\[
\dim(\mathcal{H}) = 2
\]

in the computational basis \(\{|0\rangle, |1\rangle\}\).

It is an **effective** model:
- no explicit circuit elements,
- no explicit control line transfer functions,
- a small set of operators that is sufficient for early algorithm and solver validation.

---

## 2) Hamiltonian form

FluxPulse uses the standard “drift + controls” decomposition:

\[
H(t) = H_0 + \sum_k u_k(t)\,H_k
\]

### Drift term

Phase 1 convention:

\[
H_0 = \frac{1}{2}(\omega + \Delta)\,\sigma_z
\]

- \(\omega\): qubit angular frequency (or effective splitting, depending on your units)
- \(\Delta\): optional detuning (additional static offset)
- \(\sigma_z\): Pauli Z operator

### Control terms

Phase 1 includes the following control operators (each optional via flags):

- **DriveX** channel:
\[
H_x = \frac{1}{2}\sigma_x
\]

- **DriveY** channel:
\[
H_y = \frac{1}{2}\sigma_y
\]

- **DriveZ** channel:
\[
H_z = \frac{1}{2}\sigma_z
\]

The time dependence comes entirely from the channel coefficient \(u_k(t)\) supplied by the `PulseProgram`.

---

## 3) Noise (collapse operators)

Noise is represented by Lindblad collapse operators \(C_n\) and is simulated using MCWF trajectories.

### Amplitude damping (T1)

If `t1 > 0`, the model includes one collapse operator:

\[
C_{T1} = \sqrt{\gamma_1}\,\sigma_-,
\quad \gamma_1 = 1/T_1,
\quad \sigma_- = |0\rangle\langle 1|
\]

This produces excited-state population decay with time constant \(T_1\).

### Pure dephasing (Tϕ)

If `tphi > 0`, the model includes one collapse operator:

\[
C_{\phi} = \sqrt{\gamma_\phi/2}\,\sigma_z,
\quad \gamma_\phi = 1/T_\phi
\]

The factor of 1/2 is a convention chosen so that the coherence decay rate matches \(\gamma_\phi\) under the standard Lindblad form.

---

## 4) Parameters

In code, parameters are collected in:

- `GenericQubitParams`

Fields:

- `omega`: base qubit frequency
- `detuning`: additional static Z offset
- `t1`: amplitude damping time constant (disabled if <= 0)
- `tphi`: pure dephasing time constant (disabled if <= 0)
- `enable_drive_x`, `enable_drive_y`, `enable_drive_z`: which controls to include

---

## 5) Output object: ModelBase

`build_generic_qubit_model()` produces a `ModelBase` instance:

- `hilbert_dim = 2`
- `subsystem_dims = {2}`
- `drift_terms = [H0]`
- `control_terms = [(DriveX, Hx), (DriveY, Hy), (DriveZ, Hz)]` (subset)
- `collapse_ops = [C_T1, C_phi]` (subset)

This `ModelBase` is the common input to solvers and to the `Processor` façade.

---

## 6) Typical usage

1) Create params
2) Build the model
3) Provide a pulse program (time grid + envelopes)
4) Run MCWF or unitary evolution

Pseudo-steps:

- `GenericQubitParams p; p.omega = ...; p.t1 = ...; p.enable_drive_x = true;`
- `ModelBase m; build_generic_qubit_model(p, &m);`
- `PulseProgram prog = ...;`
- `Processor proc(m, prog, ...); proc.run(...);`

---

## 7) Notes on units

FluxPulse does not enforce physical units in Phase 1. You must ensure consistency across:

- \(\omega\), detuning
- pulse amplitudes \(u(t)\)
- time grid units
- decay times `t1`, `tphi`

A common approach is to choose “simulation units” where:
- time is in microseconds (µs)
- frequencies are in radians/µs
- then \(\omega t\) is dimensionless

---

## 8) Extending beyond Phase 1

This model is intentionally minimal.

Later phases can add:
- multi-level transmon (Duffing model)
- coupling terms for 2+ qubits
- frequency-dependent drive operators
- noise channels per-qubit (embedded into multi-qubit Hilbert space)

But the interface stays the same:
- drift terms, control terms, collapse operators.

---

## Summary

The Phase 1 **generic qubit** model is a 2-level effective model with:
- drift \(\propto \sigma_z\)
- optional X/Y/Z control operators
- optional T1 and Tϕ collapse operators

It is designed to be the simplest model that still validates the full simulation pipeline (controls → Hamiltonian → MCWF trajectories → observables).
