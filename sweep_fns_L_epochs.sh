#!/bin/bash

# ================================================================

# sweep_fns_L_epochs.sh

# Large model, fns-kf, epoch sweep at 10% data

# L is slow so we anchor at 10% and cap at 20 epochs

# ================================================================

#SBATCH --job-name="fns_L_epch"

#SBATCH --output="fns_L_epch.%j.%N.out"

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --gpus=1

#SBATCH --mem=32G

#SBATCH --account=ccu108

#SBATCH --export=ALL

#SBATCH -t 01:10:00



module purge

module load cpu/0.15.4

source ~/.bashrc

conda activate pytorch_py38_env



cd /expanse/lustre/scratch/tcahill1/temp_project/morph/morph/



python sweep_script.py \
  --model_size L \
  --dataset fns-kf/solution_0.npy \
  --dataset_name fns-kf \
  --sweep_mode epochs \
  --train_pct 10 \
  --max_epochs 20 \
  --checkpoint_every 5 \
  --rollout 20 \
  --max_ar_order 16 \
  --patience 8
