#!/bin/bash

#SBATCH --job-name="stack_swe"

#SBATCH --output="stack_swe.%j.out"

#SBATCH --partition=shared

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH -t 00:30:00



module purge

module load cpu/0.15.4



source ~/.bashrc

conda activate pytorch_py38_env



cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/datasets/swe-data



python3 -c "

import numpy as np, glob

files = sorted(glob.glob('*__data.npy'))

print(f'Found {len(files)} trajectory files')

combined = np.stack([np.load(f) for f in files], axis=0)

print(f'Combined shape: {combined.shape}')

np.save('solution_0.npy', combined)

print('Done — saved to solution_0.npy')

"
