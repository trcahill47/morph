#!/bin/bash

# ================================================================

# sweep_fns_L_fraction.sh

# Large model, fns-kf dataset, data fraction sweep

# L is slower — fewer data sizes, fewer epochs to stay under 1hr

# Estimated: 3 sizes x ~15 epochs x ~5 min/epoch = ~225 min TOTAL

# so we cap at 3 sizes and 15 epochs each ~ fits in 1hr

# ================================================================

#SBATCH --job-name="fns_L_frac"

#SBATCH --output="fns_L_frac.%j.%N.out"

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=1

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 01:30:00



module purge

module load cpu/0.15.4

source ~/.bashrc

conda activate pytorch_py38_env



cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/



python sweep_script.py \
  --model_size L \
  --dataset fns-kf/solution_0.npy \
  --dataset_name fns-kf \
  --sweep_mode fraction \
  --train_sizes 5 10 15 \
  --n_epochs 15 \
  --rollout 20 \
  --max_ar_order 16 \
  --patience 8
