cmake_minimum_required(VERSION 3.20)

project(fluxpulse LANGUAGES CXX)

# ---- Options ----
option(FLUXPULSE_ENABLE_WARNINGS "Enable common compiler warnings" ON)
option(FLUXPULSE_NO_EXCEPTIONS   "Try to compile FluxPulse without C++ exceptions" ON)
option(FLUXPULSE_ENABLE_MPI      "Build the MPI backend, MPI runner, and MPI tests" ON)

# ---- Core library ----
set(FLUXPULSE_SOURCES
  # linalg
  src/linalg/dense_op.cpp
  src/linalg/blas_gemv.cpp

  # control
  src/control/time_grid.cpp
  src/control/envelope.cpp
  src/control/pulse_program.cpp

  # io
  src/io/json_reader.cpp
  src/io/config_loader.cpp
  src/io/result_writer.cpp
  src/io/trace_writer.cpp
  src/io/pulse_program_reader.cpp

  # model
  src/model/noise/collapse_ops.cpp
  src/model/effective/generic_qubit.cpp
  src/model/effective/coupled_qubits.cpp
  src/model/effective/duffing_transmon.cpp

  # sim
  src/sim/observables.cpp
  src/sim/observable_builders.cpp
  src/sim/mcwf/rk4.cpp
  src/sim/mcwf/heff.cpp
  src/sim/mcwf/jump.cpp
  src/sim/mcwf/mcwf_solver.cpp
  src/sim/lindblad/lindblad_solver.cpp

  # device
  src/device/processor.cpp
)

add_library(fluxpulse STATIC ${FLUXPULSE_SOURCES})
target_compile_features(fluxpulse PUBLIC cxx_std_20)

target_include_directories(fluxpulse
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(FLUXPULSE_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(fluxpulse PRIVATE /W4 /permissive-)
  else()
    target_compile_options(fluxpulse PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# Disable exceptions on toolchains that support -fno-exceptions.
if(FLUXPULSE_NO_EXCEPTIONS)
  target_compile_definitions(fluxpulse PUBLIC FLUXPULSE_NO_EXCEPTIONS=1)
  if(NOT MSVC)
    target_compile_options(fluxpulse PUBLIC -fno-exceptions)
  endif()
endif()

# ---- App ----
add_executable(fluxpulse_run
  apps/fluxpulse_run.cpp
)
target_link_libraries(fluxpulse_run PRIVATE fluxpulse)

# ---- MPI backend ----
#
# Keep MPI separate from the core FluxPulse library:
#   fluxpulse      -> simulator/model/control/IO core, no MPI dependency
#   fluxpulse_mpi  -> MPI orchestration backend
#
# Disable MPI support with:
#   cmake -S . -B build -DFLUXPULSE_ENABLE_MPI=OFF
#
if(FLUXPULSE_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)

  set(FLUXPULSE_MPI_SOURCES
    src/mpi/mpi_context.cpp
    src/mpi/trajectory_batcher.cpp
    src/mpi/ensemble_accumulator.cpp
    src/mpi/mpi_runner.cpp
  )

  add_library(fluxpulse_mpi STATIC ${FLUXPULSE_MPI_SOURCES})
  target_compile_features(fluxpulse_mpi PUBLIC cxx_std_20)

  target_link_libraries(fluxpulse_mpi
    PUBLIC
      fluxpulse
      MPI::MPI_CXX
  )

  if(FLUXPULSE_ENABLE_WARNINGS)
    if(MSVC)
      target_compile_options(fluxpulse_mpi PRIVATE /W4 /permissive-)
    else()
      target_compile_options(fluxpulse_mpi PRIVATE -Wall -Wextra -Wpedantic)
    endif()
  endif()

  add_executable(fluxpulse_run_mpi
    apps/fluxpulse_run_mpi.cpp
  )
  target_link_libraries(fluxpulse_run_mpi PRIVATE fluxpulse_mpi)
endif()

# ---- Tests ----
include(CTest)

if(BUILD_TESTING)

# Tests use assert() for both checks and some setup calls.
# Keep assertions active even when the library is built in Release mode.
function(fluxpulse_test_assertions exe_name)
  if(MSVC)
    target_compile_options(${exe_name} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${exe_name} PRIVATE -UNDEBUG)
  endif()
endfunction()

function(add_fluxpulse_test exe_name src_path)
  add_executable(${exe_name} ${src_path})
  target_link_libraries(${exe_name} PRIVATE fluxpulse)
  fluxpulse_test_assertions(${exe_name})
  add_test(NAME ${exe_name} COMMAND ${exe_name})
endfunction()

# Unit tests
add_fluxpulse_test(test_time_grid                     tests/unit/test_time_grid.cpp)
add_fluxpulse_test(test_envelope_eval                 tests/unit/test_envelope_eval.cpp)
add_fluxpulse_test(test_pulse_program_coeff           tests/unit/test_pulse_program_coeff.cpp)
add_fluxpulse_test(test_json_reader                   tests/unit/test_json_reader.cpp)
add_fluxpulse_test(test_config_loader                 tests/unit/test_config_loader.cpp)
add_fluxpulse_test(test_collapse_ops                  tests/unit/test_collapse_ops.cpp)
add_fluxpulse_test(test_generic_qubit_model           tests/unit/test_generic_qubit_model.cpp)
add_fluxpulse_test(test_duffing_transmon_model        tests/unit/test_duffing_transmon_model.cpp)
add_fluxpulse_test(test_rk4_unitary_constant_H        tests/unit/test_rk4_unitary_constant_H.cpp)
add_fluxpulse_test(test_mcwf_t1_decay                 tests/unit/test_mcwf_t1_decay.cpp)

add_fluxpulse_test(test_coupled_qubits_model          tests/unit/test_coupled_qubits_model.cpp)
add_fluxpulse_test(test_observable_builders           tests/unit/test_observable_builders.cpp)

add_fluxpulse_test(test_mcwf_vs_lindblad_1q_t1        tests/unit/test_mcwf_vs_lindblad_1q_t1.cpp)
add_fluxpulse_test(test_lindblad_unitary               tests/unit/test_lindblad_unitary.cpp)

add_fluxpulse_test(test_mcwf_vs_lindblad_2q_t1        tests/unit/test_mcwf_vs_lindblad_2q_t1.cpp)
add_fluxpulse_test(test_mcwf_vs_lindblad_2q_swap_noisy tests/unit/test_mcwf_vs_lindblad_2q_swap_noisy.cpp)

# Smoke tests
add_fluxpulse_test(smoketest_rng_timer                tests/smoke/smoketest_rng_timer.cpp)
add_fluxpulse_test(smoketest_core_linalg_operators    tests/smoke/smoketest_core_linalg_operators.cpp)

add_fluxpulse_test(smoketest_1q_rabi                  tests/smoke/smoketest_1q_rabi.cpp)
add_fluxpulse_test(smoketest_1q_ramsey                tests/smoke/smoketest_1q_ramsey.cpp)
add_fluxpulse_test(smoketest_1q_t1                    tests/smoke/smoketest_1q_t1.cpp)

add_fluxpulse_test(smoketest_2q_swap                  tests/smoke/smoketest_2q_swap.cpp)
add_fluxpulse_test(smoketest_2q_zz                    tests/smoke/smoketest_2q_zz.cpp)
add_fluxpulse_test(smoketest_3q_zz_chain              tests/smoke/smoketest_3q_zz_chain.cpp)

add_fluxpulse_test(smoketest_program_json_load        tests/smoke/smoketest_program_json_load.cpp)

add_fluxpulse_test(test_final_state_output             tests/unit/test_final_state_output.cpp)

# ---- MPI tests ----
if(FLUXPULSE_ENABLE_MPI)
  # EnsembleAccumulator is part of fluxpulse_mpi, but this unit test itself
  # does not need to launch multiple MPI ranks.
  add_executable(test_ensemble_accumulator
    tests/unit/test_ensemble_accumulator.cpp
  )
  target_link_libraries(test_ensemble_accumulator PRIVATE fluxpulse_mpi)
  fluxpulse_test_assertions(test_ensemble_accumulator)
  add_test(NAME test_ensemble_accumulator COMMAND test_ensemble_accumulator)

  function(add_fluxpulse_mpi_test exe_name src_path num_ranks)
    add_executable(${exe_name} ${src_path})
    target_link_libraries(${exe_name} PRIVATE fluxpulse_mpi)
    fluxpulse_test_assertions(${exe_name})

    add_test(
      NAME ${exe_name}
      COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${exe_name}>
        ${MPIEXEC_POSTFLAGS}
    )
  endfunction()

  # Two ranks are enough to exercise root/worker communication.
  add_fluxpulse_mpi_test(smoketest_mpi_1q_t1
    tests/smoke/smoketest_mpi_1q_t1.cpp
    2
  )

  add_fluxpulse_mpi_test(smoketest_mpi_2q_swap
    tests/smoke/smoketest_mpi_2q_swap.cpp
    2
  )
endif()

endif() # BUILD_TESTING
