#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=run_jax_gpu
#SBATCH -o cpu.out -e cpu.err
#SBATCH -N 1 -n 4 -t 00:10:00 --mem 10000mb -c 1
#SBATCH -p debug

python demo.py
