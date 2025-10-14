#!/bin/bash

#SBATCH --job-name=2HDM
#SBATCH --output=/home/USER/out_2HDM.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=400:00:00

module load openmpi410-gcc730
module load gcc7.3.0

mpic++ /home/USER_DIR/mpi_evolution_2HDM_Z2.cpp -o /home/USER_DIR/mpi_evolution_2HDM_Z2
mpiexec -n 64 /home/USER_DIR/mpi_evolution_2HDM_Z2