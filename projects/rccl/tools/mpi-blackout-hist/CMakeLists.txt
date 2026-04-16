# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# MPI + RCCL micro-benchmark (optional build: -DBUILD_MPI_BLACKOUT_HIST=ON)

add_executable(mpi_blackout_hist mpi_blackout_hist.cpp)

target_include_directories(mpi_blackout_hist PRIVATE "${CMAKE_BINARY_DIR}/include")

target_link_libraries(mpi_blackout_hist PRIVATE rccl MPI::MPI_CXX hip::host hip::device)
