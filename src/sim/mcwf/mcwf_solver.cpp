// src/sim/mcwf/mcwf_solver.cpp
#include "fluxpulse/sim/mcwf/mcwf_solver.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>

#include "fluxpulse/sim/observables.hpp"

namespace fluxpulse {
namespace mcwf {
namespace {

inline Real rng_u01(Rng* rng) {
  // Uses next_u64() (SplitMix64 backend) -> [0,1) using 53 bits
  const uint64_t x = rng->next_u64();
  const uint64_t mant = x >> 11;
  const double d = static_cast<double>(mant) * (1.0 / 9007199254740992.0);
  return static_cast<Real>(d);
}

inline Dim compute_num_samples(Dim n_points, Dim stride) {
  if (stride <= 0) stride = 1;
  const Dim last = n_points - 1;
  Dim count = (last / stride) + 1;
  if ((last % stride) != 0) count += 1;
  return count;
}

inline Real state_norm2(Span<const Complex> psi) {
  Real s = Real{0};
  for (Dim i = 0; i < psi.size; ++i) {
    const Complex z = psi[i];
    s += z.re * z.re + z.im * z.im;
  }
  return s;
}

Status validate_inputs(const ModelBase& model,
                       const PulseProgram& program,
                       Span<const DenseOp> observables,
                       Span<const Complex> psi0,
                       const McwfOptions& opt,
                       const McwfWorkspace& ws) {
  FLUXPULSE_RETURN_IF_ERROR(model.validate());
  FLUXPULSE_RETURN_IF_ERROR(program.grid.validate());

  const Dim dim = model.hilbert_dim;
  if (psi0.size != dim) return Error(ErrorCode::kInvalidArgument, "MCWF: psi0 size mismatch");

  if (ws.dim != dim) return Error(ErrorCode::kInvalidArgument, "MCWF: workspace dim mismatch");
  if (ws.n_obs != observables.size) return Error(ErrorCode::kInvalidArgument, "MCWF: workspace n_obs mismatch");

  for (Dim k = 0; k < observables.size; ++k) {
    if (observables[k].dim() != dim) return Error(ErrorCode::kInvalidArgument, "MCWF: observable dim mismatch");
  }

  if (opt.sample_stride <= 0) return Error(ErrorCode::kInvalidArgument, "MCWF: sample_stride must be >= 1");
  if (opt.diagnostic_stride <= 0) return Error(ErrorCode::kInvalidArgument, "MCWF: diagnostic_stride must be >= 1");
  return Ok();
}

}  // namespace

Status McwfWorkspace::init(const ModelBase& model, const PulseProgram& program, Dim n_observables) {
  FLUXPULSE_RETURN_IF_ERROR(model.validate());
  FLUXPULSE_RETURN_IF_ERROR(program.grid.validate());

  dim = model.hilbert_dim;
  n_obs = n_observables;
  n_collapse = static_cast<Dim>(model.collapse_ops.size());

  psi.resize(static_cast<size_t>(dim));
  psi_next.resize(static_cast<size_t>(dim));

  obs_tmp.resize(static_cast<size_t>(dim));
  obs_vals.resize(static_cast<size_t>(n_obs));

  k1.resize(static_cast<size_t>(dim));
  k2.resize(static_cast<size_t>(dim));
  k3.resize(static_cast<size_t>(dim));
  k4.resize(static_cast<size_t>(dim));
  rk_tmp.resize(static_cast<size_t>(dim));

  heff_tmp.resize(static_cast<size_t>(dim));
  if (n_collapse > 0) heff_tmp_k.resize(static_cast<size_t>(dim));
  else heff_tmp_k.clear();

  jump_tmp.resize(static_cast<size_t>(dim));
  jump_probs.resize(static_cast<size_t>(n_collapse));

  program_cursor = program.make_cursor();

  DenseOp K;
  FLUXPULSE_RETURN_IF_ERROR(build_cdagc_sum(model, &K));
  cdagc_sum = std::move(K);

  return Ok();
}

void McwfWorkspace::reset_cursors() {
  for (size_t i = 0; i < program_cursor.env.size(); ++i) {
    Envelope::reset(&program_cursor.env[i]);
  }
}

Status run_mcwf_trajectory(const ModelBase& model,
                          const PulseProgram& program,
                          Span<const DenseOp> observables,
                          Span<const Complex> psi0,
                          Rng* rng,
                          const McwfOptions& opt,
                          McwfWorkspace* ws,
                          Result* out) {
  if (!rng) return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory: rng is null");
  if (!ws) return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory: ws is null");
  if (!out) return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory: out is null");

  FLUXPULSE_RETURN_IF_ERROR(validate_inputs(model, program, observables, psi0, opt, *ws));

  const Dim n_points = program.grid.size();
  if (n_points <= 0) return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory: empty grid");

  const Dim stride = opt.sample_stride;
  const Dim n_samples = compute_num_samples(n_points, stride);

  out->resize(observables.size, n_samples);
  for (Dim k = 0; k < observables.size; ++k) out->observable_names[static_cast<size_t>(k)] = "";


  // Optional diagnostics (for tracing/debugging)
  McwfDiagnostics* diag = opt.diagnostics;
  const Dim diag_stride = opt.diagnostic_stride;
  Dim diag_n_samples = 0;
  Dim diag_pos = 0;
  if (diag) {
    diag_n_samples = compute_num_samples(n_points, diag_stride);
    diag->tlist.resize(static_cast<size_t>(diag_n_samples));
    diag->norm2.resize(static_cast<size_t>(diag_n_samples));
    diag->p_total.resize(static_cast<size_t>(diag_n_samples));
    diag->jumped.resize(static_cast<size_t>(diag_n_samples));
    diag->jump_index.resize(static_cast<size_t>(diag_n_samples));
    diag->jump_events.clear();
  }

  for (Dim i = 0; i < ws->dim; ++i) ws->psi[static_cast<size_t>(i)] = psi0[i];
  FLUXPULSE_RETURN_IF_ERROR(normalize_state(Span<Complex>(ws->psi.data(), ws->dim)));

  ws->program_cursor = program.make_cursor();

  HeffRhsUser rhs_user;
  rhs_user.model = &model;
  rhs_user.program = &program;
  rhs_user.program_cursor = &ws->program_cursor;

  rhs_user.ws.tmp = Span<Complex>(ws->heff_tmp.data(), ws->dim);
  if (ws->cdagc_sum.dim() != 0) {
    rhs_user.cdagc_sum = &ws->cdagc_sum;
    rhs_user.ws.tmp_k = Span<Complex>(ws->heff_tmp_k.data(), ws->dim);
  } else {
    rhs_user.cdagc_sum = nullptr;
    rhs_user.ws.tmp_k = Span<Complex>(nullptr, 0);
  }

  Rk4Workspace rk_ws;
  rk_ws.k1 = Span<Complex>(ws->k1.data(), ws->dim);
  rk_ws.k2 = Span<Complex>(ws->k2.data(), ws->dim);
  rk_ws.k3 = Span<Complex>(ws->k3.data(), ws->dim);
  rk_ws.k4 = Span<Complex>(ws->k4.data(), ws->dim);
  rk_ws.tmp = Span<Complex>(ws->rk_tmp.data(), ws->dim);

  JumpWorkspace j_ws;
  j_ws.tmp = Span<Complex>(ws->jump_tmp.data(), ws->dim);

  const Span<const CollapseOp> collapse_ops = model.collapse_ops_view();

  Dim sample_pos = 0;

  auto record_at = [&](Dim grid_i) -> Status {
    out->tlist[static_cast<size_t>(sample_pos)] = program.grid.t(grid_i);

    if (observables.size == 0) {
      sample_pos += 1;
      return Ok();
    }

    Span<const Complex> psi_span(ws->psi.data(), ws->dim);
    Span<Complex> tmp_span(ws->obs_tmp.data(), ws->dim);
    Span<Real> vals_span(ws->obs_vals.data(), observables.size);

    FLUXPULSE_RETURN_IF_ERROR(expect_many_real(observables, psi_span, tmp_span, vals_span));
    for (Dim k = 0; k < observables.size; ++k) out->expect_at(k, sample_pos) = vals_span[k];

    sample_pos += 1;
    return Ok();
  };


  auto record_diag_at = [&](Dim grid_i,
                            Real norm2_prop,
                            Real p_total_step,
                            int32_t jumped_step,
                            int32_t jump_index_step) -> Status {
    if (!diag) return Ok();
    diag->tlist[static_cast<size_t>(diag_pos)] = program.grid.t(grid_i);
    diag->norm2[static_cast<size_t>(diag_pos)] = norm2_prop;
    diag->p_total[static_cast<size_t>(diag_pos)] = p_total_step;
    diag->jumped[static_cast<size_t>(diag_pos)] = jumped_step;
    diag->jump_index[static_cast<size_t>(diag_pos)] = jump_index_step;
    diag_pos += 1;
    return Ok();
  };

  FLUXPULSE_RETURN_IF_ERROR(record_at(0));
  FLUXPULSE_RETURN_IF_ERROR(record_diag_at(0, Real{1}, Real{0}, 0, -1));

  for (Dim i = 0; i + 1 < n_points; ++i) {
    const Real t = program.grid.t(i);
    const Real dt = program.grid.dt_interval(i);

    Real p_total = Real{0};
    if (collapse_ops.size > 0) {
      FLUXPULSE_RETURN_IF_ERROR(
          compute_jump_probabilities(collapse_ops, dt,
                                     Span<const Complex>(ws->psi.data(), ws->dim),
                                     j_ws,
                                     Span<Real>(ws->jump_probs.data(), ws->n_collapse),
                                     &p_total));
    }

    FLUXPULSE_RETURN_IF_ERROR(
        rk4_step(t, dt,
                 Span<const Complex>(ws->psi.data(), ws->dim),
                 Span<Complex>(ws->psi_next.data(), ws->dim),
                 heff_rhs, &rhs_user, rk_ws));

    const Real norm2_prop = state_norm2(Span<const Complex>(ws->psi_next.data(), ws->dim));

    bool did_jump = false;
    int jump_index = -1;
    if (collapse_ops.size > 0) {
      const Real u_jump = rng_u01(rng);
      if (jump_occurs(p_total, u_jump)) {
        const Real u_pick = rng_u01(rng);
        FLUXPULSE_RETURN_IF_ERROR(
            sample_jump_index(Span<const Real>(ws->jump_probs.data(), ws->n_collapse),
                              p_total, u_pick, &jump_index));

        if (jump_index < 0 || static_cast<Dim>(jump_index) >= collapse_ops.size) {
          return Error(ErrorCode::kInternal, "run_mcwf_trajectory: invalid jump index");
        }

        FLUXPULSE_RETURN_IF_ERROR(
            apply_jump_and_normalize(collapse_ops[static_cast<Dim>(jump_index)].C,
                                     Span<Complex>(ws->psi.data(), ws->dim),
                                     j_ws, nullptr));
        did_jump = true;
        if (diag) {
          diag->jump_events.push_back(McwfJumpEvent{program.grid.t(i + 1), static_cast<int32_t>(jump_index)});
        }
      }
    }

    if (!did_jump) {
      std::swap(ws->psi, ws->psi_next);
      if (opt.renormalize_each_step) {
        FLUXPULSE_RETURN_IF_ERROR(normalize_state(Span<Complex>(ws->psi.data(), ws->dim)));
      }
    }

    const Dim ip1 = i + 1;
    if ((ip1 % diag_stride) == 0) {
      FLUXPULSE_RETURN_IF_ERROR(record_diag_at(ip1, norm2_prop, p_total,
                                             static_cast<int32_t>(did_jump ? 1 : 0),
                                             static_cast<int32_t>(jump_index)));
    }
    if ((ip1 % stride) == 0) FLUXPULSE_RETURN_IF_ERROR(record_at(ip1));
  }

  const Dim last = n_points - 1;
  if ((last % diag_stride) != 0) {
    FLUXPULSE_RETURN_IF_ERROR(record_diag_at(last, Real{1}, Real{0}, 0, -1));
  }
  if ((last % stride) != 0) FLUXPULSE_RETURN_IF_ERROR(record_at(last));

  if (diag && diag_pos != diag_n_samples) {
    return Error(ErrorCode::kInternal, "run_mcwf_trajectory: diag sample count mismatch");
  }

  if (sample_pos != n_samples) {
    return Error(ErrorCode::kInternal, "run_mcwf_trajectory: sample count mismatch");
  }

  // Optional output of the final pure state for this trajectory.
  if (opt.final_state != nullptr) {
    opt.final_state->resize(static_cast<size_t>(ws->dim));

    for (Dim i = 0; i < ws->dim; ++i) {
      (*opt.final_state)[static_cast<size_t>(i)] =
          ws->psi[static_cast<size_t>(i)];
    }
  }

  return Ok();
}

Status run_mcwf_trajectory_accumulate(const ModelBase& model,
                                     const PulseProgram& program,
                                     Span<const DenseOp> observables,
                                     Span<const Complex> psi0,
                                     Rng* rng,
                                     const McwfOptions& opt,
                                     McwfWorkspace* ws,
                                     Span<Real> sum_expectations) {
  if (!rng) return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory_accumulate: rng is null");
  if (!ws) return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory_accumulate: ws is null");

  FLUXPULSE_RETURN_IF_ERROR(validate_inputs(model, program, observables, psi0, opt, *ws));

  const Dim n_points = program.grid.size();
  const Dim stride = opt.sample_stride;
  const Dim n_samples = compute_num_samples(n_points, stride);
  const Dim n_obs = observables.size;

  if (sum_expectations.size != n_obs * n_samples) {
    return Error(ErrorCode::kInvalidArgument, "run_mcwf_trajectory_accumulate: sum_expectations size mismatch");
  }

  for (Dim i = 0; i < ws->dim; ++i) ws->psi[static_cast<size_t>(i)] = psi0[i];
  FLUXPULSE_RETURN_IF_ERROR(normalize_state(Span<Complex>(ws->psi.data(), ws->dim)));
  ws->program_cursor = program.make_cursor();

  HeffRhsUser rhs_user;
  rhs_user.model = &model;
  rhs_user.program = &program;
  rhs_user.program_cursor = &ws->program_cursor;

  rhs_user.ws.tmp = Span<Complex>(ws->heff_tmp.data(), ws->dim);
  if (ws->cdagc_sum.dim() != 0) {
    rhs_user.cdagc_sum = &ws->cdagc_sum;
    rhs_user.ws.tmp_k = Span<Complex>(ws->heff_tmp_k.data(), ws->dim);
  } else {
    rhs_user.cdagc_sum = nullptr;
    rhs_user.ws.tmp_k = Span<Complex>(nullptr, 0);
  }

  Rk4Workspace rk_ws;
  rk_ws.k1 = Span<Complex>(ws->k1.data(), ws->dim);
  rk_ws.k2 = Span<Complex>(ws->k2.data(), ws->dim);
  rk_ws.k3 = Span<Complex>(ws->k3.data(), ws->dim);
  rk_ws.k4 = Span<Complex>(ws->k4.data(), ws->dim);
  rk_ws.tmp = Span<Complex>(ws->rk_tmp.data(), ws->dim);

  JumpWorkspace j_ws;
  j_ws.tmp = Span<Complex>(ws->jump_tmp.data(), ws->dim);

  const Span<const CollapseOp> collapse_ops = model.collapse_ops_view();

  Dim sample_pos = 0;

  auto record_into_sum = [&]() -> Status {
    if (n_obs == 0) {
      sample_pos += 1;
      return Ok();
    }

    Span<const Complex> psi_span(ws->psi.data(), ws->dim);
    Span<Complex> tmp_span(ws->obs_tmp.data(), ws->dim);
    Span<Real> vals_span(ws->obs_vals.data(), n_obs);

    FLUXPULSE_RETURN_IF_ERROR(expect_many_real(observables, psi_span, tmp_span, vals_span));

    const Dim nT = n_samples;
    for (Dim k = 0; k < n_obs; ++k) {
      sum_expectations[static_cast<size_t>(k) * static_cast<size_t>(nT) +
                       static_cast<size_t>(sample_pos)] += vals_span[k];
    }

    sample_pos += 1;
    return Ok();
  };

  FLUXPULSE_RETURN_IF_ERROR(record_into_sum());

  for (Dim i = 0; i + 1 < n_points; ++i) {
    const Real t = program.grid.t(i);
    const Real dt = program.grid.dt_interval(i);

    Real p_total = Real{0};
    if (collapse_ops.size > 0) {
      FLUXPULSE_RETURN_IF_ERROR(
          compute_jump_probabilities(collapse_ops, dt,
                                     Span<const Complex>(ws->psi.data(), ws->dim),
                                     j_ws,
                                     Span<Real>(ws->jump_probs.data(), ws->n_collapse),
                                     &p_total));
    }

    FLUXPULSE_RETURN_IF_ERROR(
        rk4_step(t, dt,
                 Span<const Complex>(ws->psi.data(), ws->dim),
                 Span<Complex>(ws->psi_next.data(), ws->dim),
                 heff_rhs, &rhs_user, rk_ws));

    bool did_jump = false;
    if (collapse_ops.size > 0) {
      const Real u_jump = rng_u01(rng);
      if (jump_occurs(p_total, u_jump)) {
        const Real u_pick = rng_u01(rng);
        int jidx = -1;
        FLUXPULSE_RETURN_IF_ERROR(
            sample_jump_index(Span<const Real>(ws->jump_probs.data(), ws->n_collapse),
                              p_total, u_pick, &jidx));
        if (jidx < 0 || static_cast<Dim>(jidx) >= collapse_ops.size) {
          return Error(ErrorCode::kInternal, "run_mcwf_trajectory_accumulate: invalid jump index");
        }
        FLUXPULSE_RETURN_IF_ERROR(
            apply_jump_and_normalize(collapse_ops[static_cast<Dim>(jidx)].C,
                                     Span<Complex>(ws->psi.data(), ws->dim),
                                     j_ws, nullptr));
        did_jump = true;
      }
    }

    if (!did_jump) {
      std::swap(ws->psi, ws->psi_next);
      if (opt.renormalize_each_step) {
        FLUXPULSE_RETURN_IF_ERROR(normalize_state(Span<Complex>(ws->psi.data(), ws->dim)));
      }
    }

    const Dim ip1 = i + 1;
    if ((ip1 % stride) == 0) FLUXPULSE_RETURN_IF_ERROR(record_into_sum());
  }

  const Dim last = n_points - 1;
  if ((last % stride) != 0) FLUXPULSE_RETURN_IF_ERROR(record_into_sum());

  if (sample_pos != n_samples) {
    return Error(ErrorCode::kInternal, "run_mcwf_trajectory_accumulate: sample count mismatch");
  }
  return Ok();
}

}  // namespace mcwf
}  // namespace fluxpulse