# Duffing Transmon Effective Model (FluxPulse)

FluxPulse includes a **Duffing transmon** effective model to represent a multi-level superconducting qubit with anharmonic level spacing, using a truncated oscillator basis.

This model is a natural next step beyond the 2-level `generic_qubit` model because it can capture **leakage** into higher excited states and supports more realistic drive operators.

---

## 1) Hilbert space

The model uses a truncated Fock basis:

\[
\{|0\rangle, |1\rangle, \dots, |N-1\rangle\}
\]

with truncation dimension:

- `levels = N` (must be \(\ge 2\))

So:

\[
\dim(\mathcal{H}) = N
\]

---

## 2) Drift Hamiltonian (Duffing oscillator)

The drift Hamiltonian is diagonal in the number basis and uses the Duffing form:

\[
H_0 = \omega\,n + \frac{1}{2}\alpha\,n(n-1)
\]

where:

- \(n\) is the number operator:
  \[
  n = \mathrm{diag}(0,1,2,\dots,N-1)
  \]
- \(\omega\) is the fundamental transition frequency scale
- \(\alpha\) is the anharmonicity (often negative for transmons)

Energy levels are:

\[
E_m = \omega m + \frac{1}{2}\alpha m(m-1)
\]

so the transition frequency between \(|m\rangle\to|m+1\rangle\) is:

\[
E_{m+1} - E_m = \omega + \alpha m
\]

This non-uniform spacing is what makes a transmon “qubit-like” when driven near the 0→1 transition.

---

## 3) Control Hamiltonians (drive operators)

FluxPulse uses the standard ladder operators in the truncated basis:

### Annihilation and creation operators
\[
\langle m|a|m+1\rangle = \sqrt{m+1},
\qquad
a^\dagger = (a)^{\dagger}
\]

### DriveX and DriveY controls

The model supports the following control operators (each optional):

- **DriveX**:
\[
H_x = s\,(a + a^\dagger)
\]

- **DriveY**:
\[
H_y = s\,i\,(a^\dagger - a)
\]

where `s = drive_scale` is a user-chosen scaling factor that maps envelope amplitude to Hamiltonian strength.

The full time-dependent Hamiltonian is then:

\[
H(t) = H_0 + u_x(t) H_x + u_y(t) H_y
\]

where \(u_x(t), u_y(t)\) are the classical coefficients from the `PulseProgram` channels.

---

## 4) Noise (collapse operators)

Noise is represented via Lindblad collapse operators and is typically simulated using MCWF trajectories.

### A) Relaxation / energy decay (T1)

If `t1 > 0`, the model includes:

\[
C_{T1} = \sqrt{\gamma_1}\,a,
\qquad \gamma_1 = 1/T_1
\]

This produces decay down the ladder (\(|m\rangle \to |m-1\rangle\)) with rates proportional to \(m\).

### B) Dephasing (Tϕ)

If `tphi > 0`, the model includes:

\[
C_{\phi} = \sqrt{\gamma_\phi/2}\,n,
\qquad \gamma_\phi = 1/T_\phi
\]

This is a convenient diagonal dephasing operator in the number basis.

> Note: dephasing conventions can differ by factors; FluxPulse uses the same “/2” convention as the qubit pure-dephasing helper to stay internally consistent.

---

## 5) Parameters

In code, parameters are provided via:

- `DuffingTransmonParams`

Fields:

- `levels`: truncation dimension \(N\)
- `omega`: drift frequency scale
- `anharm`: anharmonicity \(\alpha\)
- `drive_scale`: multiplier for drive operators
- `t1`: relaxation time constant (<=0 disables)
- `tphi`: dephasing time constant (<=0 disables)
- `enable_drive_x`, `enable_drive_y`: which channels to expose

---

## 6) Output object: ModelBase

`build_duffing_transmon_model()` returns a `ModelBase` with:

- `hilbert_dim = levels`
- `subsystem_dims = {levels}`
- `drift_terms = [H0]`
- `control_terms = [(DriveX, Hx), (DriveY, Hy)]` (subset)
- `collapse_ops = [C_T1, C_phi]` (subset)

Solvers only depend on `ModelBase`, so the Duffing model can be swapped in without changing solver code.

---

## 7) Usage notes

### Choosing truncation `levels`
A small truncation is often enough for early tests:

- `levels = 3` for basic leakage detection
- `levels = 4–6` for more realistic strong-drive studies

Higher truncations increase cost roughly as \(O(N^2)\) per matrix-vector application when using dense operators.

### Units
FluxPulse does not enforce units. Ensure consistency across:
- `omega`, `anharm`, drive amplitudes `u(t)`
- time grid units
- `t1`, `tphi`

A common choice is:
- time in microseconds (µs)
- frequencies in radians/µs

---

## 8) Extensions (future)

Possible Phase 2+ improvements:
- rotating-frame formulation (drive at carrier frequency)
- explicit charge dispersion / flux dependence
- thermal excitation and photon shot-noise dephasing models
- coupling to resonators and cross-Kerr terms
- multi-transmon coupling operators embedded into larger Hilbert spaces

---

## Summary

The Duffing transmon model in FluxPulse is a truncated multi-level anharmonic oscillator with:

- drift: \(H_0 = \omega n + \tfrac{1}{2}\alpha n(n-1)\)
- controls: DriveX \(\propto (a+a^\dagger)\), DriveY \(\propto i(a^\dagger-a)\)
- optional noise: relaxation \(C\propto a\), dephasing \(C\propto n\)

It provides a more realistic “qubit” model than a strict 2-level system, enabling leakage and multi-level effects while still fitting cleanly into the `ModelBase` abstraction.
