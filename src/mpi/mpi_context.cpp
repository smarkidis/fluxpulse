// src/mpi/mpi_context.cpp
#include "fluxpulse/mpi/mpi_context.hpp"

#include <mpi.h>

namespace fluxpulse::mpi {
namespace {

inline bool mpi_is_initialized() noexcept {
  int flag = 0;
  MPI_Initialized(&flag);
  return flag != 0;
}

inline bool mpi_is_finalized() noexcept {
  int flag = 0;
  MPI_Finalized(&flag);
  return flag != 0;
}

}  // namespace

MpiContext::MpiContext(int& argc, char**& argv) {
  const bool was_initialized = mpi_is_initialized();
  if (!was_initialized) {
    MPI_Init(&argc, &argv);
    owns_mpi_ = true;
  }

  if (mpi_is_initialized() && !mpi_is_finalized()) {
    initialized_ = true;

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rank_ = static_cast<Index>(rank);
    size_ = static_cast<Index>(size);
  } else {
    initialized_ = false;
    owns_mpi_ = false;
    rank_ = static_cast<Index>(0);
    size_ = static_cast<Index>(1);
  }
}

MpiContext::~MpiContext() {
  if (!owns_mpi_) return;
  if (!mpi_is_initialized()) return;
  if (mpi_is_finalized()) return;

  MPI_Finalize();
}

void MpiContext::Barrier() const {
  if (!initialized_) return;
  if (mpi_is_finalized()) return;

  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace fluxpulse::mpi