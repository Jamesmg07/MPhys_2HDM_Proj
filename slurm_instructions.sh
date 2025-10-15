#!/bin/bash

#SBATCH --job-name=2HDM
#SBATCH --output=/share/centaurus_nas/mkza/2HDM_out.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=400:00:00

module load openmpi410-gcc730
module load gcc7.3.0

mpic++ ./monopole_antimonopole.cpp -o ./Executables/monopole_antimonopole
mpiexec -n 32 ./Executables/monopole_antimonopole

