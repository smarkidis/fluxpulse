# Channels in FluxPulse (Phase 1)

In FluxPulse, a **channel** is a named (but runtime-efficient) handle for a *classical control signal* that modulates the quantum device in time.

A channel answers the question:

> “Which time-dependent knob is being applied, and which Hamiltonian term does it multiply?”

Channels are **not** quantum degrees of freedom. They are **classical control lines** (drive/flux/coupler/etc.) represented as time-dependent real-valued coefficients.

---

## The mathematical role of a channel

FluxPulse represents the (possibly time-dependent) system Hamiltonian as:

\[
H(t) = H_0 + \sum_k u_k(t)\,H_k
\]

- **Drift**: \(H_0\) — time-independent part (e.g., qubit frequency terms, static couplings).
- **Control terms**: \(H_k\) — fixed operators (matrices) that represent how a given control acts on the device.
- **Channel coefficient**: \(u_k(t)\) — the *envelope* value at time \(t\) for channel \(k\).

A **channel** is the identifier that links:
1) the coefficient function \(u_k(t)\) (pulse/envelope), and  
2) the operator \(H_k\) (control Hamiltonian term).

---

## Software view: ChannelId and PulseProgram

### ChannelId
A channel is represented by a small **struct** identifier:

- `ChannelId` is a lightweight, strongly-typed key (internally a small integer).
- The simulator core uses `ChannelId` instead of strings to keep inner loops fast and allocation-free.

Example (Phase 1 convention):
- `ChannelId{1}` → DriveX
- `ChannelId{2}` → DriveY
- `ChannelId{3}` → DriveZ

### PulseProgram
A `PulseProgram` is the schedule of classical controls over a `TimeGrid`.

Conceptually:
- A `PulseProgram` contains a list of `(ChannelId, Envelope)` entries.
- At a given time `t`, the program can answer: **what is the coefficient** `u(t)` for a particular `ChannelId`?

So the pipeline is:
- **Config / code** defines pulse envelopes per channel  
→ `PulseProgram` evaluates `u_k(t)`  
→ solver builds/apply `H(t)` using the model’s `H_k` for the same channel  
→ integrates the state (unitary or MCWF).

---

## Examples

### 1-qubit (Phase 1)
A typical single-qubit effective model might define:
- Drift: `H0 = 0.5 * ω * Z`
- Controls:
  - `DriveX` channel multiplies `Hx = 0.5 * X`
  - `DriveY` channel multiplies `Hy = 0.5 * Y`

Then the Hamiltonian is:
\[
H(t) = 0.5\,\omega Z + u_x(t)\,0.5X + u_y(t)\,0.5Y
\]

### Multi-qubit (later Phase 1 / Phase 2 direction)
A channel may represent:
- a microwave drive line targeting a specific qubit,
- a flux line changing a qubit frequency,
- a coupler flux controlling interaction strength,
- a bias line used for tuning.

In more advanced setups, you may want the identifier to encode *both* a channel type and a target (e.g., “DriveX on qubit 2”).  
For Phase 1 we keep it minimal and can build that targeting logic in the model/config layer.

---

## Channels vs pulses vs envelopes

- **Channel**: “which knob/control line” (key/ID).
- **Pulse**: “a scheduled use of a channel” (channel + envelope).
- **Envelope**: “the function over time” (piecewise constant, piecewise linear, etc.).

Multiple pulses could conceptually be summed on the same channel, but in Phase 1 we usually keep **one envelope per channel** in a program (easy to reason about and trace).

---

## Trace / visualization (Perfetto)

When exporting a pulse timeline:
- Each **channel** maps naturally to a separate **track** (e.g., `tid = channel.value`).
- Each envelope segment (e.g., a piecewise-constant segment) becomes a time slice event on that track.
- This makes it easy to see overlap and sequencing across channels.

---

## Practical conventions (recommended)

1) **Reserve `ChannelId{0}` as invalid**.
2) Keep channel IDs stable and documented.
3) Use **strings only in config/I/O**, then map to `ChannelId` once.
4) Treat channel coefficients `u(t)` as **real-valued** unless you explicitly add IQ modeling (later).

---

## Summary

A FluxPulse **channel** is a stable identifier for a classical control coefficient \(u_k(t)\) that multiplies a fixed operator \(H_k\) in the Hamiltonian.  
It’s the glue between:
- the **control schedule** (PulseProgram), and
- the **physics model** (control Hamiltonian terms),
and it’s also the natural unit of organization for debug traces and visualization.
