#!/bin/bash

## administrative things
#SBATCH --job-name solve_nelson
#SBATCH -A AST24021

## necessary compute power
#SBATCH --partition gg
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 14
#SBATCH --time 00:01:00

## output
#SBATCH -o output_file.log
#SBATCH -e error_file.log

## actual commands
module load gcc
module load openmpi
module load cuda
module load python3_mpi

ibrun python3 generate_nelson_data_TACC.py "input_p_small.npy"

