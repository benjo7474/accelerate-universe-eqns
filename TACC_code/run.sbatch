#!/bin/bash

## administrative things
#SBATCH --job-name solve_nelson
#SBATCH -A AST24021

## necessary compute power
#SBATCH --partition gg
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 144
#SBATCH --time 00:01:00

## output
#SBATCH --output "slurm_output.log"
#SBATCH --error "slurm_error.log"

## actual commands
module load gcc/15.1.0
module load openmpi/5.0.7
module load cuda/12.6
module load python3_mpi/3.11.8

ibrun python3 generate_nelson_data_TACC.py "input_p_small.npy" 9

