// include/fluxpulse/mpi/mpi_context.hpp
#pragma once

#include "fluxpulse/core/types.hpp"

namespace fluxpulse::mpi {

// Thin RAII wrapper for MPI runtime lifetime and MPI_COMM_WORLD metadata.
// Keep <mpi.h> out of this header; include it only in the .cpp.
class MpiContext {
 public:
  MpiContext(int& argc, char**& argv);
  ~MpiContext();

  MpiContext(const MpiContext&) = delete;
  MpiContext& operator=(const MpiContext&) = delete;
  MpiContext(MpiContext&&) = delete;
  MpiContext& operator=(MpiContext&&) = delete;

  bool valid() const noexcept { return initialized_; }

  Index rank() const noexcept { return rank_; }
  Index size() const noexcept { return size_; }
  bool is_root() const noexcept { return rank_ == 0; }

  void Barrier() const;

 private:
  bool initialized_ = false;
  bool owns_mpi_ = false;

  Index rank_ = 0;
  Index size_ = 1;
};

}  // namespace fluxpulse::mpi