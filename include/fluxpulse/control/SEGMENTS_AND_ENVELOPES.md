# Segments and Envelopes in FluxPulse (Phase 1)

FluxPulse represents time-dependent classical control signals (pulse shapes) using **envelopes** built from **segments**.

This document explains the concepts and the Phase 1 conventions used in the codebase.

---

## Why segments and envelopes?

Control coefficients in FluxPulse appear in the Hamiltonian as:

\[
H(t) = H_0 + \sum_k u_k(t)\,H_k
\]

- \(H_0\): drift Hamiltonian (time-independent)
- \(H_k\): fixed control operators
- \(u_k(t)\): time-dependent real coefficient for channel \(k\)

FluxPulse models each \(u_k(t)\) as an **Envelope**: a function of time assembled from simple **Segments**.

This has three advantages:

1) **Determinism and clarity**: the pulse program is a small, explicit schedule.  
2) **Performance**: evaluation during the solver loop is cheap and allocation-free.  
3) **Traceability**: segments map naturally to timeline slices for visualization (e.g., Perfetto).

---

## Segment

A **Segment** is a local rule for the envelope value on a time interval.

### Interval convention: half-open

Each segment is active on the **half-open interval**:

\[
[t_0, t_1)
\]

Meaning:

- active for \(t_0 \le t < t_1\)
- inactive for \(t < t_0\) and \(t \ge t_1\)

Half-open intervals make boundaries unambiguous when segments abut exactly:
- Segment A ends at \(t_1\)
- Segment B begins at \(t_1\)
- Then \(t = t_1\) belongs to **B**, not A.

### Segment types in Phase 1

FluxPulse Phase 1 supports two segment kinds:

#### 1) PiecewiseConstant (ZOH)
A constant value over the interval:

\[
u(t) = v_0 \quad \text{for } t \in [t_0, t_1)
\]

This is also called **ZOH**: **Zero-Order Hold**, meaning the signal is “held” constant until the next update.

Parameters:
- `t0`, `t1`
- `v0` (constant value)

#### 2) PiecewiseLinear
Linear interpolation between endpoint values:

\[
u(t_0) = v_0, \quad u(t_1) = v_1
\]

and for \(t \in [t_0, t_1)\):

\[
u(t) = v_0 + \alpha(t)\,(v_1 - v_0), \quad \alpha(t)=\frac{t-t_0}{t_1-t_0}
\]

Parameters:
- `t0`, `t1`
- `v0` (value at `t0`)
- `v1` (value at `t1`)

---

## Envelope

An **Envelope** is an ordered list of segments defining a complete control coefficient function \(u(t)\).

### Phase 1 conventions

FluxPulse uses these conventions for envelopes in Phase 1:

1) **Sorted by start time**: segments are ordered by `t0`.
2) **Non-overlapping**: segments must not overlap in time. Formally, if `prev` and `next` are consecutive:
   - `prev.t1 <= next.t0`
   - gaps are allowed
3) **Outside all segments: value is 0**:
   - if no segment contains `t`, then `u(t) = 0`

These conventions keep evaluation simple and remove ambiguity about “what if two segments overlap?”

### Example envelope with a gap

Segments:

- constant `0.3` from `0.0` to `1.0`
- gap from `1.0` to `2.0` (implicit 0)
- linear ramp from `2.0` to `3.0` from `0.0` to `0.5`

Then:

- `u(t)=0.3` for `0<=t<1`
- `u(t)=0` for `1<=t<2`
- `u(t)` ramps from `0` to `0.5` for `2<=t<3`

---

## Evaluation strategies

FluxPulse provides two evaluation approaches:

### 1) `eval(t)` (binary search)
Looks up the segment containing time `t` using a search over sorted segments.
This is reliable for random access times but is \(O(\log N)\).

### 2) `eval_monotonic(t, cursor)` (recommended for solver loops)
Solver time steps typically advance forward in time.
`eval_monotonic` uses a small **Cursor** (segment index) and advances it as time increases, giving **amortized O(1)** evaluation.

This matches the performance rules (no allocations in hot path; simple loops).

---

## Relationship to pulses and channels

- A **Channel** identifies *which control knob* is being driven.
- A **Pulse** is typically: `(ChannelId, Envelope)`.
- A **PulseProgram** groups pulses and a time grid, and can answer:
  - “What is the coefficient `u_k(t)` for channel `k` at time `t`?”

Segments and envelopes are the representation used for those coefficients.

---

## Tracing and visualization

Because segments are explicit time intervals, they map naturally to timeline events:

- each channel becomes a separate track (`tid`)
- each segment becomes a slice (`ts = t0`, `dur = t1 - t0`, `args = { value(s) }`)

This makes it easy to debug pulse schedules visually.

---

## Summary

- A **Segment** defines how `u(t)` behaves on `[t0, t1)`.
- An **Envelope** is a set of non-overlapping segments, sorted by time, with `u(t)=0` outside segments.
- PiecewiseConstant (ZOH) and PiecewiseLinear are enough for Phase 1 tests (Rabi/Ramsey/T1), while remaining compatible with later, more complex pulse modeling.
