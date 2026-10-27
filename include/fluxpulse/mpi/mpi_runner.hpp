// include/fluxpulse/mpi/mpi_runner.hpp
#pragma once

#include <vector>

#include "fluxpulse/core/error.hpp"
#include "fluxpulse/core/span.hpp"
#include "fluxpulse/core/types.hpp"
#include "fluxpulse/mpi/ensemble_accumulator.hpp"
#include "fluxpulse/mpi/mpi_context.hpp"
#include "fluxpulse/mpi/mpi_types.hpp"

namespace fluxpulse {

struct Complex;
struct Processor;
struct Result;

namespace mpi {

// Inputs for one MPI run.
// Keep this thin: Processor already packages model/program/observables/options.
struct MpiRunInputs {
  const Processor* processor = nullptr;
  Span<const Complex> psi0;
  MpiRunConfig mpi;
};

// Lightweight metadata that can be useful for logging/debug output.
struct MpiRunSummary {
  TrajectoryId total_trajectories = 0;
  TrajectoryId local_trajectories = 0;

  Index observable_count = 0;
  Index sample_count = 0;

  SchedulingMode scheduling = SchedulingMode::kDynamic;
};

class MpiRunner {
 public:
  explicit MpiRunner(const MpiContext* ctx) : ctx_(ctx) {}

  MpiRunner(const MpiRunner&) = delete;
  MpiRunner& operator=(const MpiRunner&) = delete;

  // Run the MCWF ensemble across MPI ranks.
  //
  // Root rank:
  // - orchestrates work
  // - aggregates all partial statistics
  // - finalizes Result
  //
  // Non-root ranks:
  // - participate in the run
  // - do not finalize/write output
  Status Run(const MpiRunInputs& in,
             Result* out_result,
             MpiRunSummary* out_summary = nullptr) const;

  // Execute one local trajectory batch and accumulate into `acc`.
  //
  // This mirrors the serial Processor multi-trajectory path, but only for the
  // assigned half-open range:
  //
  //   [assignment.traj_begin, assignment.traj_begin + assignment.traj_count)
  //
  Status RunLocalBatch(const Processor& proc,
                       Span<const Complex> psi0,
                       const BatchAssignment& assignment,
                       EnsembleAccumulator* acc) const;

  // Infer result tensor shape using Processor's model/program/options
  // conventions.
  Status InferResultShape(const Processor& proc,
                          Index* out_observable_count,
                          Index* out_sample_count) const;

  // Build sampled time points exactly like Processor::run_mcwf().
  Status BuildSampleTimes(const Processor& proc,
                          std::vector<Real>* out_tlist) const;

 private:
  Status RunStatic(const MpiRunInputs& in,
                   Result* out_result,
                   MpiRunSummary* out_summary) const;

  Status RunDynamic(const MpiRunInputs& in,
                    Result* out_result,
                    MpiRunSummary* out_summary) const;

  // Root-side only: convert accumulated sum/sum_sq into mean/var/stderr Result.
  Status FinalizeRootResult(const Processor& proc,
                            const EnsembleAccumulator& global_acc,
                            Result* out_result) const;

  const MpiContext* ctx_ = nullptr;
};

}  // namespace mpi
}  // namespace fluxpulse
