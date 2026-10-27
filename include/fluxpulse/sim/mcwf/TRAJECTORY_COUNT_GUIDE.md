# Choosing the Number of Trajectories in MCWF / Quantum-Jump Simulations

This note explains **how to pick the number of Monte‑Carlo trajectories** `Ntraj` for MCWF (Monte‑Carlo Wave Function / quantum‑jump) simulations, with **theory**, **practical rules of thumb**, and **tables of `Ntraj` vs. target error**.

> TL;DR  
> If you want a **95% confidence half‑width** (roughly “mean ± ε”) for a bounded observable, a safe estimate is  
> \[
> N_{\text{traj}} \gtrsim \frac{(1.96)^2\,\mathrm{Var}_{\max}}{\varepsilon^2}
> \]
> with conservative bounds: **Var≤1** for Pauli‑type observables in \([-1,1]\), and **Var≤0.25** for projectors/probabilities in \([0,1]\).  
> For many “final plot quality” results, **2k–10k** trajectories is common; for fast iteration, **64–512** is often enough.

---

## 1. What MCWF estimates (and why you need many trajectories)

MCWF produces **stochastic trajectories** \(\psi^{(m)}(t)\) with random jump events.  
For an observable \(O\), each trajectory yields a sample value:
\[
X_m(t)=\langle \psi^{(m)}(t) | O | \psi^{(m)}(t)\rangle.
\]

The MCWF estimator of the Lindblad expectation \(\mathbb{E}[X(t)]\) is the sample mean:
\[
\overline{X}(t)=\frac{1}{N}\sum_{m=1}^N X_m(t).
\]

The estimation error is statistical and decays as \(1/\sqrt{N}\).

---

## 2. Standard error, variance, and confidence intervals

Let \(\mathrm{Var}[X(t)]\) be the (population) variance across trajectories at time \(t\).  
Then the **standard error of the mean** is:
\[
\mathrm{stderr}(t)=\sqrt{\frac{\mathrm{Var}[X(t)]}{N}}.
\]

Under the central limit theorem (typically a good approximation for moderate/large \(N\)), a **95% confidence interval** is approximately:
\[
\overline{X}(t)\pm 1.96\,\mathrm{stderr}(t).
\]

### How to turn a target error into `Ntraj`
If you want a **95% half‑width** \(\varepsilon\) (so “±ε”), solve:
\[
1.96\,\sqrt{\frac{\mathrm{Var}_{\max}}{N}} \le \varepsilon
\quad\Rightarrow\quad
N \ge \frac{(1.96)^2\,\mathrm{Var}_{\max}}{\varepsilon^2}.
\]

Here \(\mathrm{Var}_{\max}\) should be the maximum variance over the **time region you care about**.

---

## 3. Conservative variance bounds (when you don’t know Var)

Many observables are bounded, giving worst‑case variance bounds you can use before running anything.

### 3.1 Projectors / probabilities \([0,1]\)
For \(X\in[0,1]\):  
\[
\mathrm{Var}[X]\le \tfrac{1}{4}.
\]
This is tight (achieved by a 50/50 mixture of 0 and 1).

### 3.2 Pauli expectation values \([-1,1]\)
For \(X\in[-1,1]\):  
\[
\mathrm{Var}[X]\le 1.
\]

Use these bounds to choose a *safe* starting point.

---

## 4. Tables: `Ntraj` needed for common error targets

Assume a **95% CI half‑width** \(\varepsilon\), i.e. “mean ± ε”.

### 4.1 Pauli‑like observables (Var≤1)
\[
N \gtrsim \frac{3.84}{\varepsilon^2}\quad(\text{since }1.96^2\approx 3.84)
\]

| ε (95% half‑width) | Ntraj (Var=1 worst‑case) |
|---:|---:|
| 0.10 | 384 |
| 0.05 | 1,536 |
| 0.02 | 9,600 |
| 0.01 | 38,400 |
| 0.005 | 153,600 |

### 4.2 Projectors / probabilities (Var≤0.25)
\[
N \gtrsim \frac{0.96}{\varepsilon^2}
\]

| ε (95% half‑width) | Ntraj (Var=0.25 worst‑case) |
|---:|---:|
| 0.10 | 96 |
| 0.05 | 384 |
| 0.02 | 2,400 |
| 0.01 | 9,600 |
| 0.005 | 38,400 |

> These are conservative. Real runs often have smaller variance for much of the timeline.

---

## 5. Best practice workflow (recommended)

### Step A — Run a cheap pilot
Run with a smallish \(N\) (e.g. **256**), compute/record:
- `var(t)` or `stderr(t)` per observable (your simulator writes these as `expect_var` / `expect_stderr`)

### Step B — Decide the time window and observable(s)
You rarely need tight error bars everywhere. Choose:
- which observables matter (e.g., populations vs. \(Z\) vs. correlators)
- which time region matters (e.g., around a gate, around a revival, final time)

Compute the **maximum** `stderr(t)` over that region, call it `stderr_old_max`.

### Step C — Scale `Ntraj`
Because \(\mathrm{stderr}\propto 1/\sqrt{N}\):
\[
N_{\text{new}} \approx N_{\text{old}}\left(\frac{\mathrm{stderr}_{\text{old,max}}}{\mathrm{stderr}_{\text{target}}}\right)^2.
\]

This is the fastest way to pick `Ntraj` realistically.

---

## 6. Worked example (typical)
Suppose your pilot run uses \(N=256\) and you see a peak variance of about:
\[
\mathrm{Var}_{\max}\approx 0.84
\]
for some \(Z\)-observable during the interesting part of the timeline.

Then peak stderr is roughly:
\[
\sqrt{0.84/256}\approx 0.057.
\]

If your target is **stderr ≈ 0.02** (95% band ≈ ±0.04), you need:
\[
N\approx 0.84/0.02^2\approx 2{,}100.
\]

If your target is **stderr ≈ 0.01** (95% band ≈ ±0.02), you need:
\[
N\approx 0.84/0.01^2\approx 8{,}400.
\]

---

## 7. Relative error caveat (don’t get fooled)
A “relative stderr” like \(\mathrm{stderr}/|\overline{X}|\) can explode when the mean crosses near 0.  
That often *looks* scary but may be fine in absolute terms.

**Prefer absolute error bars** (stderr, or ±1.96·stderr), unless you are specifically validating a quantity expected to stay far from 0.

---

## 8. Practical trajectory count recommendations

These assume you want reasonably smooth curves, not publication‑grade confidence bands everywhere.

- **Debugging / quick iteration:** 64–512  
  (fast, catches bugs, see qualitative trends)
- **Routine comparisons / regression checks:** 1k–5k  
  (stable enough to detect solver changes)
- **Final plots / tight error bars:** 5k–50k  
  (depends on how tight ε is and on the variance peaks)

If you have MPI batching later, it becomes cheap to push toward the higher end.

---

## 9. Notes on correlated sampling across time
Each trajectory generates a full time series, so samples across time are correlated **within** a trajectory.
That does **not** break the variance/stderr formulas at each time \(t\), because those are computed **across trajectories at fixed \(t\)**.

---

## 10. Checklist for choosing `Ntraj`
- [ ] Run a pilot (e.g., `N=256`) with `expect_stderr` enabled.
- [ ] Identify observables + time window of interest.
- [ ] Take max(stderr) in that window.
- [ ] Choose target stderr or 95% half‑width.
- [ ] Scale `N` using \(N\propto 1/\varepsilon^2\).
- [ ] Re-run at the new `N` and confirm stderr meets target.

---

## Appendix: quick formulas

- stderr at time \(t\):  
  \[
  \mathrm{stderr}(t)=\sqrt{\frac{\mathrm{Var}[X(t)]}{N}}
  \]
- 95% CI (approx):  
  \[
  \overline{X}(t)\pm 1.96\,\mathrm{stderr}(t)
  \]
- Required `N` for 95% half‑width ε:  
  \[
  N \ge \frac{(1.96)^2\mathrm{Var}_{\max}}{\varepsilon^2}
  \]
- Conservative bounds:  
  \[
  \mathrm{Var}\le 1\;(\text{Pauli}),\qquad \mathrm{Var}\le 0.25\;(\text{projector})
  \]
