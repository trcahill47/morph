#!/bin/bash

# ================================================================

# sweep_fns_M_fraction.sh

# Medium model, fns-kf dataset, data fraction sweep

# Trains to 20 epochs at each of 4 data sizes

# Estimated time: ~20 runs x ~3 min/epoch = ~60 min

# ================================================================

#SBATCH --job-name="fns_M_frac"

#SBATCH --output="fns_M_frac.%j.%N.out"

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=1

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 01:00:00



module purge

module load cpu/0.15.4

source ~/.bashrc

conda activate pytorch_py38_env



cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/



python sweep_script.py \
  --model_size M \
  --dataset fns-kf/solution_0.npy \
  --dataset_name fns-kf \
  --sweep_mode fraction \
  --train_sizes 5 10 15 20 \
  --n_epochs 20 \
  --rollout 20 \
  --patience 8
