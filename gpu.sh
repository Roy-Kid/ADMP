#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=run_jax_gpu
#SBATCH -o gpu.out -e gpu.err
#SBATCH -N 1 -n 1 -t 00:10:00 --mem 10000mb -c 1
#SBATCH --gres=gpu:1 -p gpu

python test_jax.py
