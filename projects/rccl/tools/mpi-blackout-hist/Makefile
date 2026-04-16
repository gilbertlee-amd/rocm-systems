MPI_DIR=/opt/sre-tools/ompi
RCCL_DIR=../../build/release

mpi_blackout_hist: mpi_blackout_hist.cpp
	/opt/rocm/bin/hipcc mpi_blackout_hist.cpp -o mpi_blackout_hist -I$(MPI_DIR)/include -I/$(RCCL_DIR)/include -L$(MPI_DIR)/lib -L$(RCCL_DIR) -lmpi -lrccl

test: mpi_blackout_hist
	$(MPI_DIR)/bin/mpirun -np 8 mpi_blackout_hist

clean:
	rm -f ./mpi_blackout_hist
