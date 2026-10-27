// src/mpi/mpi_runner.cpp
#include "fluxpulse/mpi/mpi_runner.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/rng.hpp"
#include "fluxpulse/device/processor.hpp"
#include "fluxpulse/mpi/trajectory_batcher.hpp"
#include "fluxpulse/sim/mcwf/mcwf_solver.hpp"
#include "fluxpulse/sim/result.hpp"

namespace fluxpulse::mpi {
namespace {

constexpr int kTagRequest    = 1001;
constexpr int kTagAssign     = 1002;
constexpr int kTagTrajCount  = 1003;
constexpr int kTagSum        = 1004;
constexpr int kTagSumSq      = 1005;

// Match Processor / MCWF sampling convention:
// sample at 0, stride, 2*stride, ... and always include the last point.
inline Dim compute_num_samples(Dim n_points, Dim stride) {
  if (n_points <= 0) return 0;
  if (stride <= 0) stride = 1;
  const Dim last = n_points - 1;
  Dim n = last / stride + 1;
  if ((last % stride) != 0) n += 1;
  return n;
}

inline Status check_mpi(int code, const char* where) {
  if (code == MPI_SUCCESS) return Ok();
  return Error(ErrorCode::kInternal, where);
}

inline Status checked_byte_count(std::size_t n, int* out) {
  if (!out) return Error(ErrorCode::kInvalidArgument, "checked_byte_count: out is null");
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Error(ErrorCode::kInvalidArgument, "checked_byte_count: byte count exceeds MPI int limit");
  }
  *out = static_cast<int>(n);
  return Ok();
}

inline Status send_bytes(const void* data, std::size_t nbytes, int dst, int tag) {
  int count = 0;
  FLUXPULSE_RETURN_IF_ERROR(checked_byte_count(nbytes, &count));
  FLUXPULSE_RETURN_IF_ERROR(check_mpi(
      MPI_Send(const_cast<void*>(data), count, MPI_BYTE, dst, tag, MPI_COMM_WORLD),
      "MPI_Send failed"));
  return Ok();
}

inline Status recv_bytes(void* data, std::size_t nbytes, int src, int tag) {
  int count = 0;
  FLUXPULSE_RETURN_IF_ERROR(checked_byte_count(nbytes, &count));
  MPI_Status st{};
  FLUXPULSE_RETURN_IF_ERROR(check_mpi(
      MPI_Recv(data, count, MPI_BYTE, src, tag, MPI_COMM_WORLD, &st),
      "MPI_Recv failed"));
  return Ok();
}

inline Status send_request(Index rank) {
  BatchRequest req{};
  req.worker_rank = rank;
  return send_bytes(&req, sizeof(req), 0, kTagRequest);
}

inline Status send_assignment(const BatchAssignment& a, int dst) {
  return send_bytes(&a, sizeof(a), dst, kTagAssign);
}

inline Status recv_assignment(BatchAssignment* a) {
  if (!a) return Error(ErrorCode::kInvalidArgument, "recv_assignment: a is null");
  return recv_bytes(a, sizeof(*a), 0, kTagAssign);
}

inline Status send_partial_result(const EnsembleAccumulator& acc, int dst) {
  const TrajectoryId n_tr = acc.trajectories();
  FLUXPULSE_RETURN_IF_ERROR(send_bytes(&n_tr, sizeof(n_tr), dst, kTagTrajCount));

  const Span<const Real> sum = acc.sum();
  const Span<const Real> sum_sq = acc.sum_sq();
  const std::size_t nbytes = static_cast<std::size_t>(sum.size) * sizeof(Real);

  FLUXPULSE_RETURN_IF_ERROR(send_bytes(sum.data, nbytes, dst, kTagSum));
  FLUXPULSE_RETURN_IF_ERROR(send_bytes(sum_sq.data, nbytes, dst, kTagSumSq));
  return Ok();
}

inline Status recv_partial_result(int src, EnsembleAccumulator* tmp) {
  if (!tmp) return Error(ErrorCode::kInvalidArgument, "recv_partial_result: tmp is null");

  TrajectoryId n_tr = 0;
  FLUXPULSE_RETURN_IF_ERROR(recv_bytes(&n_tr, sizeof(n_tr), src, kTagTrajCount));

  Span<Real> sum = tmp->sum();
  Span<Real> sum_sq = tmp->sum_sq();
  const std::size_t nbytes = static_cast<std::size_t>(sum.size) * sizeof(Real);

  FLUXPULSE_RETURN_IF_ERROR(recv_bytes(sum.data, nbytes, src, kTagSum));
  FLUXPULSE_RETURN_IF_ERROR(recv_bytes(sum_sq.data, nbytes, src, kTagSumSq));

  tmp->set_trajectories(n_tr);
  return Ok();
}

}  // namespace

Status MpiRunner::InferResultShape(const Processor& proc,
                                   Index* out_observable_count,
                                   Index* out_sample_count) const {
  if (!out_observable_count) {
    return Error(ErrorCode::kInvalidArgument, "MpiRunner::InferResultShape: out_observable_count is null");
  }
  if (!out_sample_count) {
    return Error(ErrorCode::kInvalidArgument, "MpiRunner::InferResultShape: out_sample_count is null");
  }

  FLUXPULSE_RETURN_IF_ERROR(proc.validate());

  const Dim n_obs = static_cast<Dim>(proc.observables.size());
  const Dim n_points = proc.program.grid.size();
  const Dim stride = (proc.opt.sample_stride <= 0) ? 1 : proc.opt.sample_stride;
  const Dim n_samples = compute_num_samples(n_points, stride);

  *out_observable_count = n_obs;
  *out_sample_count = n_samples;
  return Ok();
}

Status MpiRunner::BuildSampleTimes(const Processor& proc,
                                   std::vector<Real>* out_tlist) const {
  if (!out_tlist) {
    return Error(ErrorCode::kInvalidArgument, "MpiRunner::BuildSampleTimes: out_tlist is null");
  }

  FLUXPULSE_RETURN_IF_ERROR(proc.validate());

  const Dim n_points = proc.program.grid.size();
  const Dim stride = (proc.opt.sample_stride <= 0) ? 1 : proc.opt.sample_stride;
  const Dim n_samples = compute_num_samples(n_points, stride);

  out_tlist->assign(static_cast<std::size_t>(n_samples), Real{0});
  for (Dim i = 0; i < n_samples; ++i) {
    Dim grid_i = i * stride;
    if (grid_i >= n_points) grid_i = n_points - 1;
    (*out_tlist)[static_cast<std::size_t>(i)] = proc.program.grid.t(grid_i);
  }
  return Ok();
}

Status MpiRunner::RunLocalBatch(const Processor& proc,
                                Span<const Complex> psi0,
                                const BatchAssignment& assignment,
                                EnsembleAccumulator* acc) const {
  if (!acc) return Error(ErrorCode::kInvalidArgument, "MpiRunner::RunLocalBatch: acc is null");
  FLUXPULSE_RETURN_IF_ERROR(proc.validate());

  const Dim d = proc.model.hilbert_dim;
  if (psi0.size != d) {
    return Error(ErrorCode::kInvalidArgument, "MpiRunner::RunLocalBatch: psi0 dimension mismatch");
  }

  if (assignment.terminate || assignment.traj_count <= 0) {
    acc->Clear();
    return Ok();
  }

  Index n_obs = 0;
  Index n_samples = 0;
  FLUXPULSE_RETURN_IF_ERROR(InferResultShape(proc, &n_obs, &n_samples));

  if (acc->observable_count() != n_obs || acc->sample_count() != n_samples) {
    acc->Reset(n_obs, n_samples);
  } else {
    acc->Clear();
  }

  mcwf::McwfWorkspace ws;
  FLUXPULSE_RETURN_IF_ERROR(ws.init(proc.model, proc.program, n_obs));

  mcwf::McwfOptions mopt;
  mopt.sample_stride = proc.opt.sample_stride;
  mopt.renormalize_each_step = proc.opt.renormalize_each_step;
  mopt.diagnostic_stride = 1;
  mopt.diagnostics = nullptr;  // ensemble path: no diagnostics

  Result traj;
  traj.resize(n_obs, n_samples);

  for (TrajectoryId tr = assignment.traj_begin;
       tr < assignment.traj_begin + static_cast<TrajectoryId>(assignment.traj_count);
       ++tr) {
    const Seed seed = derive_trajectory_seed(proc.opt.base_seed, tr);

    Rng rng;
    rng.init_splitmix64();
    rng.seed(seed);

    FLUXPULSE_RETURN_IF_ERROR(
        mcwf::run_mcwf_trajectory(
            proc.model,
            proc.program,
            Span<const DenseOp>(proc.observables.data(), static_cast<Dim>(proc.observables.size())),
            psi0,
            &rng,
            mopt,
            &ws,
            &traj));

    acc->AccumulateTrajectory(
        Span<const Real>(traj.expectations.data(),
                         static_cast<Dim>(traj.expectations.size())));
  }

  return Ok();
}

Status MpiRunner::FinalizeRootResult(const Processor& proc,
                                     const EnsembleAccumulator& global_acc,
                                     Result* out_result) const {
  if (!out_result) {
    return Error(ErrorCode::kInvalidArgument, "MpiRunner::FinalizeRootResult: out_result is null");
  }

  if (!global_acc.FinalizeInto(out_result)) {
    return Error(ErrorCode::kInternal, "MpiRunner::FinalizeRootResult: FinalizeInto failed");
  }

  std::vector<Real> tlist;
  FLUXPULSE_RETURN_IF_ERROR(BuildSampleTimes(proc, &tlist));
  out_result->tlist = std::move(tlist);

  if (!proc.observable_names.empty()) {
    out_result->observable_names = proc.observable_names;
  }

  return Ok();
}

Status MpiRunner::Run(const MpiRunInputs& in,
                      Result* out_result,
                      MpiRunSummary* out_summary) const {
  if (!ctx_) return Error(ErrorCode::kInvalidArgument, "MpiRunner::Run: ctx is null");
  if (!ctx_->valid()) return Error(ErrorCode::kInvalidArgument, "MpiRunner::Run: MPI context is not valid");
  if (!in.processor) return Error(ErrorCode::kInvalidArgument, "MpiRunner::Run: processor is null");
  FLUXPULSE_RETURN_IF_ERROR(in.processor->validate());

  if (ctx_->is_root() && !out_result) {
    return Error(ErrorCode::kInvalidArgument, "MpiRunner::Run: out_result is null on root");
  }

  const SchedulingMode sched = TrajectoryBatcher::EffectiveScheduling(in.mpi);
  if (sched == SchedulingMode::kStatic) {
    return RunStatic(in, out_result, out_summary);
  }
  return RunDynamic(in, out_result, out_summary);
}

Status MpiRunner::RunStatic(const MpiRunInputs& in,
                            Result* out_result,
                            MpiRunSummary* out_summary) const {
  const Processor& proc = *in.processor;

  const TrajectoryId total_traj =
      (in.mpi.total_trajectories > 0) ? in.mpi.total_trajectories
                                      : static_cast<TrajectoryId>(proc.opt.trajectories);

  Index n_obs = 0;
  Index n_samples = 0;
  FLUXPULSE_RETURN_IF_ERROR(InferResultShape(proc, &n_obs, &n_samples));

  const BatchAssignment local_assignment =
      TrajectoryBatcher::StaticAssignment(total_traj, ctx_->size(), ctx_->rank());

  EnsembleAccumulator local_acc(n_obs, n_samples);
  FLUXPULSE_RETURN_IF_ERROR(RunLocalBatch(proc, in.psi0, local_assignment, &local_acc));

  if (out_summary) {
    out_summary->total_trajectories = total_traj;
    out_summary->local_trajectories =
        local_assignment.terminate ? static_cast<TrajectoryId>(0)
                                   : static_cast<TrajectoryId>(local_assignment.traj_count);
    out_summary->observable_count = n_obs;
    out_summary->sample_count = n_samples;
    out_summary->scheduling = SchedulingMode::kStatic;
  }

  if (ctx_->is_root()) {
    EnsembleAccumulator global_acc(n_obs, n_samples);
    global_acc.Merge(local_acc);

    EnsembleAccumulator tmp(n_obs, n_samples);
    for (Index src = 1; src < ctx_->size(); ++src) {
      tmp.Clear();
      FLUXPULSE_RETURN_IF_ERROR(recv_partial_result(static_cast<int>(src), &tmp));
      global_acc.Merge(tmp);
    }

    FLUXPULSE_RETURN_IF_ERROR(FinalizeRootResult(proc, global_acc, out_result));
  } else {
    FLUXPULSE_RETURN_IF_ERROR(send_partial_result(local_acc, 0));
  }

  ctx_->Barrier();
  return Ok();
}

Status MpiRunner::RunDynamic(const MpiRunInputs& in,
                             Result* out_result,
                             MpiRunSummary* out_summary) const {
  const Processor& proc = *in.processor;

  const TrajectoryId total_traj =
      (in.mpi.total_trajectories > 0) ? in.mpi.total_trajectories
                                      : static_cast<TrajectoryId>(proc.opt.trajectories);
  const Index batch_size = TrajectoryBatcher::NormalizeBatchSize(in.mpi.batch_size, 1);

  Index n_obs = 0;
  Index n_samples = 0;
  FLUXPULSE_RETURN_IF_ERROR(InferResultShape(proc, &n_obs, &n_samples));

  if (out_summary) {
    out_summary->total_trajectories = total_traj;
    out_summary->local_trajectories = 0;
    out_summary->observable_count = n_obs;
    out_summary->sample_count = n_samples;
    out_summary->scheduling = SchedulingMode::kDynamic;
  }

  // Single-rank fallback: dynamic batching locally.
  if (ctx_->size() == 1) {
    EnsembleAccumulator global_acc(n_obs, n_samples);
    EnsembleAccumulator local_acc(n_obs, n_samples);

    TrajectoryId next_traj = 0;
    while (true) {
      const BatchAssignment a =
          TrajectoryBatcher::NextDynamicBatch(total_traj, batch_size, &next_traj);
      if (a.terminate) break;

      FLUXPULSE_RETURN_IF_ERROR(RunLocalBatch(proc, in.psi0, a, &local_acc));
      global_acc.Merge(local_acc);

      if (out_summary) {
        out_summary->local_trajectories += static_cast<TrajectoryId>(a.traj_count);
      }
    }

    FLUXPULSE_RETURN_IF_ERROR(FinalizeRootResult(proc, global_acc, out_result));
    return Ok();
  }

  if (ctx_->is_root()) {
    EnsembleAccumulator global_acc(n_obs, n_samples);
    EnsembleAccumulator tmp(n_obs, n_samples);

    TrajectoryId next_traj = 0;
    Index live_workers = ctx_->size() - 1;

    while (live_workers > 0) {
      MPI_Status st{};
      FLUXPULSE_RETURN_IF_ERROR(check_mpi(
          MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st),
          "MPI_Probe failed"));

      if (st.MPI_TAG == kTagRequest) {
        BatchRequest req{};
        FLUXPULSE_RETURN_IF_ERROR(recv_bytes(&req, sizeof(req), st.MPI_SOURCE, kTagRequest));

        const BatchAssignment a =
            TrajectoryBatcher::NextDynamicBatch(total_traj, batch_size, &next_traj);

        FLUXPULSE_RETURN_IF_ERROR(send_assignment(a, st.MPI_SOURCE));
        if (a.terminate) {
          live_workers -= 1;
        }
      } else if (st.MPI_TAG == kTagTrajCount) {
        tmp.Clear();
        FLUXPULSE_RETURN_IF_ERROR(recv_partial_result(st.MPI_SOURCE, &tmp));
        global_acc.Merge(tmp);
      } else {
        return Error(ErrorCode::kInternal, "MpiRunner::RunDynamic: unexpected MPI tag");
      }
    }

    FLUXPULSE_RETURN_IF_ERROR(FinalizeRootResult(proc, global_acc, out_result));
  } else {
    EnsembleAccumulator local_acc(n_obs, n_samples);

    while (true) {
      FLUXPULSE_RETURN_IF_ERROR(send_request(ctx_->rank()));

      BatchAssignment a{};
      FLUXPULSE_RETURN_IF_ERROR(recv_assignment(&a));
      if (a.terminate) break;

      FLUXPULSE_RETURN_IF_ERROR(RunLocalBatch(proc, in.psi0, a, &local_acc));
      FLUXPULSE_RETURN_IF_ERROR(send_partial_result(local_acc, 0));

      if (out_summary) {
        out_summary->local_trajectories += static_cast<TrajectoryId>(a.traj_count);
      }
    }
  }

  ctx_->Barrier();
  return Ok();
}

}  // namespace fluxpulse::mpi